"""Runtime bootstrap for Jarvis.

`build_runtime(input_dir)` performs all of the one-time startup work the RAG
pipeline needs — configure the LLM and embedding model, load and cache the
document/image corpus, build (or load) the persistent vector index, construct
the hybrid retriever, and attach a TTS manager — then returns a ready-to-use
`QueryRuntime`. Keeping this out of `Jarvis.py` means importing the entry script
no longer triggers model loads or indexing as a side effect.
"""

import os

# Let unsupported MPS ops fall back to CPU instead of erroring. torch reads this
# at import, so it must be set before llama_index pulls torch in below.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from jarvisrag.config import JARVIS_DOMAIN, short_text_qa_template
from jarvisrag.embeddings import bound_mps_embedding_shapes
from jarvisrag.speech import TTSManager, _SilentTTS
from jarvisrag.speech.chatterbox import ChatterboxTTSClient, _CLIENT_LOG
from jarvisrag.extraction.extractors import (
    file_extractor,
    has_files_with_extensions,
    load_documents_with_cache,
)
from jarvisrag.vectorstore import create_milvus_vector_store
from jarvisrag.extraction.indexing import (
    build_or_load_persistent_index,
    filter_informative_documents,
    group_documents_by_source_windows,
)
from jarvisrag.retrieval import build_hybrid_retriever
from jarvisrag.iotools import Spinner, _is_port_open
from jarvisrag.mcp.connector import load_mcp_registry
from jarvisrag.docsync import load_document_sources
from jarvisrag.answer.answer import QueryRuntime


PERSIST_DIR = os.path.abspath(os.path.expanduser(os.environ.get(
    "JARVIS_PERSIST_DIR",
    os.path.join(os.path.dirname(__file__), "jarvis_storage"),
)))
EXTRACTION_CACHE_DIR = os.path.join(PERSIST_DIR, "extraction_cache")

# Measured on the retrieval evaluation set: widening past this added candidate
# noise that cost ranking quality and latency without improving recall.
DEFAULT_RETRIEVAL_TOP_K = "32"

DOCUMENT_EXTENSIONS = [
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".ppt", ".pptx", ".txt", ".csv", ".rtf",
    ".html", ".htm", ".mhtml", ".mht", ".emlx",
]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"]


def _build_llm() -> Ollama:
    # make sure ollama is running: `ollama run llama3`
    # JARVIS_LLM_MODEL selects the primary synthesis model (deployment/hardware
    # choice, mirrors JARVIS_ESCALATION_MODEL). Default is the MoE build: it matched
    # the dense 31B on the answer harness (identical accuracy) at ~4x faster median
    # latency; the dense tag stays selectable via the env var.
    model = os.environ.get("JARVIS_LLM_MODEL", "gemma4:26b-mlx").strip() or "gemma4:26b-mlx"
    kwargs: dict = dict(
        model=model,
        request_timeout=600,
        temperature=0.2,
        additional_kwargs={"num_predict": 256},
        # Ollama 0.35 turned thinking on by default for gemma4; the hidden trace
        # consumes the short verdict budgets (review PASS/FAIL, triage) and the
        # empty reply is read as a pass. Answers were validated without it.
        thinking=False,
    )
    # Loaded context window. The model's own max is 262144 (256K). The largest
    # possible prompt here is bounded by the evidence caps (30 lines x 8000 chars
    # + MCP + generation) at ~100k tokens worst case, ~20k typical, so 131072
    # (128K, ChatGPT-class) never truncates while halving KV-cache memory vs 256K.
    # Raise toward 262144 for more headroom; 0 keeps the model's own default.
    try:
        num_ctx = int(os.environ.get("JARVIS_NUM_CTX", "131072"))
    except ValueError:
        num_ctx = 131072
    if num_ctx > 0:
        kwargs["context_window"] = num_ctx
        kwargs["additional_kwargs"] = {"num_predict": 256, "num_ctx": num_ctx}
    return Ollama(**kwargs)


def _auto_embed_device() -> str:
    """Best available torch device for embeddings (hardware choice).

    Apple Silicon (MPS) and CUDA are used automatically when present, else CPU.
    ``JARVIS_EMBED_DEVICE`` overrides for a specific deployment."""
    override = os.environ.get("JARVIS_EMBED_DEVICE", "").strip().lower()
    if override:
        return override
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _build_embed_model() -> HuggingFaceEmbedding:
    # Local embedding model — bge-m3: 8192-token context, 1024-dim, multilingual,
    # strong retrieval ranking on heterogeneous content (resumes, contracts, PDFs).
    # Do NOT add query/text instructions — bge-m3 is not prefix-trained and adding
    # them degrades retrieval quality.
    # embed_batch_size raised from the default (10) so bulk indexing embeds more
    # chunks per forward pass.
    device = _auto_embed_device()
    print(f"[jarvis] Embedding model device: {device}")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", embed_batch_size=32, device=device)
    bound_mps_embedding_shapes(embed_model)
    return embed_model


def _load_corpus_documents(input_dir: str) -> list:
    """Load document-type files and prepend a summary document listing them."""
    if not has_files_with_extensions(input_dir, DOCUMENT_EXTENSIONS):
        print("\nNo documents found in directory")
        return []

    with Spinner("Processing documents"):
        corpus_docs = load_documents_with_cache(
            input_dir=input_dir,
            required_exts=DOCUMENT_EXTENSIONS,
            file_extractor=file_extractor,
            cache_dir=EXTRACTION_CACHE_DIR,
            recursive=True,
        )

    # Track ALL files in input directory that match extensions (even if they produce no docs)
    all_attempted_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in DOCUMENT_EXTENSIONS):
                all_attempted_files.append(os.path.join(root, file))

    # Get unique file paths from extracted documents
    all_extracted_file_paths = sorted(
        {str(doc.metadata.get("file_path")) for doc in corpus_docs if doc.metadata.get("file_path")}
    )

    # Create comprehensive summary: files processed + files attempted
    summary_lines = [
        "COMPLETE FILE LIST - ALL PROCESSED DOCUMENTS",
        f"Total files found: {len(all_attempted_files)}",
        f"Total documents extracted: {len(all_extracted_file_paths)}",
        "",
        "FILENAMES:",
    ]
    for path in all_extracted_file_paths:
        summary_lines.append(f"- {os.path.basename(path)}")

    summary_lines.extend(["", "FULL PATHS:"])
    summary_lines.extend(f"- {path}" for path in all_extracted_file_paths)

    summary_doc = Document(
        text="\n".join(summary_lines),
        metadata={"type": "file_list", "file_path": "__summary__", "source": "summary", "is_file_list": True},
    )
    corpus_docs.insert(0, summary_doc)
    print(
        f"Added summary document for {len(all_extracted_file_paths)} extracted files "
        f"from {len(all_attempted_files)} total files"
    )
    return corpus_docs


def _load_image_documents(input_dir: str, llm) -> tuple[list, object | None]:
    """Load image-type files. Returns (image_docs, optional image chat engine)."""
    if not has_files_with_extensions(input_dir, IMAGE_EXTENSIONS):
        print("\nNo image files found in directory")
        return [], None

    img_chat_engine = None
    try:
        with Spinner("Processing images"):
            img_docs = load_documents_with_cache(
                input_dir=input_dir,
                required_exts=IMAGE_EXTENSIONS,
                file_extractor=file_extractor,
                cache_dir=EXTRACTION_CACHE_DIR,
                recursive=True,
            )
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
        return img_docs, img_chat_engine
    except Exception as e:
        print(f"Warning: Error processing images: {e}")
        return [], None


def _build_index(all_docs: list, embed_model, persist_dir=None, collection_name=None, *, owned_corpus_dir=None):
    """Filter/group documents and build or load the persistent vector index.

    Returns (doc_index, storage_context, response_mode). `all_docs` is mutated to
    the filtered/grouped set actually indexed. ``persist_dir``/``collection_name``
    override the module defaults so a domain corpus can persist into its own
    Milvus collection + directory and reload instead of re-embedding."""
    persist_dir = persist_dir or PERSIST_DIR
    raw_doc_count = len(all_docs)
    if owned_corpus_dir is None:
        with Spinner(f"Filtering {raw_doc_count} documents"):
            all_docs[:] = filter_informative_documents(all_docs)
    filtered_doc_count = raw_doc_count - len(all_docs)
    if filtered_doc_count:
        print(f"Filtered {filtered_doc_count} low-information extracted chunks before indexing.")
    if not all_docs:
        raise RuntimeError("All extracted chunks were low-information; nothing remains to index.")

    context_mode = os.environ.get("JARVIS_CONTEXT_MODE", "chunk").strip().lower()
    if context_mode not in {"chunk", "source_window"}:
        raise ValueError("JARVIS_CONTEXT_MODE must be 'chunk' or 'source_window'.")
    if owned_corpus_dir is not None and context_mode != "chunk":
        raise ValueError("Checkpointed mail indexing currently requires JARVIS_CONTEXT_MODE=chunk.")
    if context_mode == "source_window":
        context_window_words = int(os.environ.get("JARVIS_CONTEXT_WINDOW_WORDS", "2500"))
        extracted_doc_count = len(all_docs)
        all_docs[:] = group_documents_by_source_windows(all_docs, max_words=context_window_words)
        print(
            f"Grouped {extracted_doc_count} extracted chunks into {len(all_docs)} "
            f"source-window context documents (max_words={context_window_words})."
        )

    vector_backend = os.environ.get("JARVIS_VECTOR_BACKEND", "milvus").strip().lower()
    if vector_backend not in {"milvus", "local", "memory"}:
        raise ValueError("JARVIS_VECTOR_BACKEND must be 'milvus', 'local', or 'memory'.")
    if owned_corpus_dir is not None and vector_backend == "memory":
        raise ValueError("Checkpointed mail indexing requires a persistent local or Milvus backend.")

    use_milvus = vector_backend == "milvus"
    require_milvus = use_milvus and os.environ.get("JARVIS_REQUIRE_MILVUS", "1") == "1"
    sync_existing_index = os.environ.get("JARVIS_SYNC_EXISTING_INDEX", "0") == "1"
    rebuild_index = os.environ.get("JARVIS_REBUILD_INDEX", "0") == "1"
    response_mode = os.environ.get("JARVIS_RESPONSE_MODE", "query").strip().lower()
    if response_mode not in {"query", "chat"}:
        raise ValueError("JARVIS_RESPONSE_MODE must be 'query' or 'chat'.")

    milvus_host = os.environ.get("MILVUS_HOST", "127.0.0.1")
    milvus_port = os.environ.get("MILVUS_PORT", "19530")
    milvus_collection = collection_name or os.environ.get("MILVUS_COLLECTION", "jarvis_context")

    if filtered_doc_count and vector_backend != "memory" and not rebuild_index:
        print(
            "Warning: low-information chunks were filtered from the current extraction, "
            "but an existing index may still contain old vectors. Jarvis will rebuild automatically "
            "when the index manifest is missing or stale; set JARVIS_REBUILD_INDEX=1 to force a rebuild."
        )

    active_vector_backend = "memory" if vector_backend == "memory" else "local"
    doc_index = None
    storage_context = None

    print(
        f"Vector backend -> {vector_backend}. "
        f"Milvus configuration -> enabled={use_milvus}, required={require_milvus}, "
        f"host={milvus_host}, port={milvus_port}, collection={milvus_collection}"
    )

    if vector_backend == "memory":
        print("Using in-memory VectorStoreIndex.from_documents backend (pre-Milvus style).")
        doc_index = VectorStoreIndex.from_documents(all_docs, embed_model=embed_model)
        active_vector_backend = "memory"

    if vector_backend != "memory":
        if use_milvus and not _is_port_open(milvus_host, milvus_port, timeout=2.0):
            if require_milvus:
                raise RuntimeError(
                    f"Milvus is required but unreachable at {milvus_host}:{milvus_port}. "
                    "Start Milvus or set JARVIS_REQUIRE_MILVUS=0 to allow local fallback."
                )
            print(f"Warning: Cannot reach Milvus at {milvus_host}:{milvus_port}. Skipping Milvus and using local fallback.")
            use_milvus = False
        elif use_milvus:
            print(f"Milvus TCP health check passed at {milvus_host}:{milvus_port}.")

        try:
            if use_milvus:
                vector_store = create_milvus_vector_store(
                    collection_name=milvus_collection,
                    host=milvus_host,
                    port=milvus_port,
                )
                try:
                    has_collection = vector_store.has_collection()
                    print(f"Milvus collection '{milvus_collection}' exists={has_collection}")
                except Exception as e:
                    print(f"Warning: Unable to verify Milvus collection state: {e}")
            else:
                from llama_index.core.vector_stores.simple import SimpleVectorStore
                vector_store = SimpleVectorStore()

            print("Building or loading persistent index...")
            doc_index, storage_context = build_or_load_persistent_index(
                all_docs,
                vector_store,
                persist_dir=persist_dir,
                embed_model=embed_model,
                sync_existing=sync_existing_index,
                rebuild=rebuild_index,
                **({"owned_corpus_dir": owned_corpus_dir, "owned_extensions": (".emlx",)}
                   if owned_corpus_dir is not None else {}),
            )
            if use_milvus:
                active_vector_backend = "milvus"
                print(f"Using Milvus vector store at {milvus_host}:{milvus_port}/{milvus_collection}")
            else:
                active_vector_backend = "local"
                print("Using local vector store.")
        except Exception as e:
            if owned_corpus_dir is not None:
                raise RuntimeError("Mail index verification failed; checkpoints retained for retry.") from e
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
                persist_dir=persist_dir,
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
    return doc_index, storage_context, response_mode, active_vector_backend


def _build_tts():
    """Construct the TTS manager, or a silent stub when TTS is disabled."""
    if os.environ.get("JARVIS_ENABLE_TTS", "1") != "1":
        return _SilentTTS()
    if os.environ.get("JARVIS_TTS_ENGINE", "kokoro").strip().lower() == "chatterbox":
        return _build_chatterbox_tts()
    return TTSManager()


def _build_chatterbox_tts():
    """Voice-cloned engine running under its own interpreter, Kokoro on failure.

    Chatterbox cannot share Jarvis's environment (it pins transformers/torch
    versions that conflict), so it needs an external interpreter and a checkout;
    when either is unconfigured this stays on Kokoro rather than failing.
    """
    chatterbox_dir = os.environ.get("JARVIS_CHATTERBOX_DIR", "").strip()
    python = os.environ.get("JARVIS_CHATTERBOX_PYTHON", "").strip()
    ref_wav = os.environ.get("JARVIS_CHATTERBOX_REF_WAV", "").strip()

    if chatterbox_dir:
        python = python or os.path.join(chatterbox_dir, ".venv", "bin", "python")
    if not python or not ref_wav:
        print(
            "[jarvis] JARVIS_TTS_ENGINE=chatterbox needs JARVIS_CHATTERBOX_PYTHON "
            "(or JARVIS_CHATTERBOX_DIR) and JARVIS_CHATTERBOX_REF_WAV; using Kokoro."
        )
        return TTSManager()

    worker = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts",
        "chatterbox_tts_worker.py",
    )

    def _optional_float(name):
        raw = os.environ.get(name, "").strip()
        try:
            return float(raw) if raw else None
        except ValueError:
            return None

    silence_ms = os.environ.get("JARVIS_CHATTERBOX_SILENCE_MS", "").strip()
    print(
        f"[tts] Speech engine -> chatterbox (voice: {os.path.basename(ref_wav)}); "
        f"loading in the background. Speech logs: {_CLIENT_LOG}"
    )
    return ChatterboxTTSClient(
        python,
        worker,
        ref_wav,
        exaggeration=_optional_float("JARVIS_CHATTERBOX_EXAGGERATION"),
        cfg_weight=_optional_float("JARVIS_CHATTERBOX_CFG_WEIGHT"),
        silence_ms=int(silence_ms) if silence_ms.isdigit() else None,
        fallback_factory=TTSManager,
    )


def _partition_home_files(input_dir: str, indexed_docs: list) -> tuple[list[str], list[str]]:
    """Split the launch folder's files into (indexed, excluded) basenames.

    `indexed` = files that produced at least one chunk that survived extraction +
    informativeness filtering; `excluded` = files that physically exist but
    yielded no usable chunk (e.g. unreadable scans where OCR found no text). Both
    are returned so the References block can list every provided-folder file —
    indexed ones plainly, excluded ones annotated — regardless of whether the LLM
    happened to cite them (filenames with spaces are unreliable in prose)."""
    supported = tuple(ext.lower() for ext in (DOCUMENT_EXTENSIONS + IMAGE_EXTENSIONS))
    present: dict[str, str] = {}
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(supported):
                fpath = os.path.join(root, fname)
                present[os.path.abspath(fpath).lower()] = os.path.basename(fpath)
    indexed_keys: set[str] = set()
    for doc in indexed_docs:
        meta = getattr(doc, "metadata", None) or {}
        fpath = str(meta.get("file_path") or "")
        if fpath and fpath != "__summary__":
            indexed_keys.add(os.path.abspath(fpath).lower())
    items = sorted(present.items(), key=lambda kv: kv[1].lower())
    indexed = [name for key, name in items if key in indexed_keys]
    excluded = [name for key, name in items if key not in indexed_keys]
    return indexed, excluded


def _build_domain_retrievers(embed_model, llm) -> list:
    """Build gated, SEPARATE retrievers for volatile domain corpora (e.g.
    federal-benefits PDFs) declared in document_sources.json.

    Each domain persists into its OWN collection + directory, so it has its own
    docstore (BM25 cannot leak into or crowd the main index, no persist/collection
    collision with production) AND it reloads on later startups instead of
    re-embedding. A domain is consulted only when a query matches its routing
    terms (handled in the answer layer). Active whenever a set with match_any and
    a present corpus_dir exists; any per-set failure is logged and skipped."""
    try:
        sets = load_document_sources()
    except Exception as exc:
        print(f"[jarvis] Domain-corpus layer: could not load document_sources.json ({exc}); skipping.")
        return []

    top_k = int(os.environ.get("JARVIS_RETRIEVAL_TOP_K", DEFAULT_RETRIEVAL_TOP_K))
    domains: list = []
    for doc_set in sets:
        if not doc_set.match_any or not os.path.isdir(doc_set.corpus_dir):
            continue
        # Persist each domain into its OWN collection + directory: isolated
        # docstore (no BM25 leak, no collision with the main index) that reloads
        # on later startups instead of re-embedding.
        safe = "".join(c if c.isalnum() else "_" for c in doc_set.name)
        domain_persist = os.path.join(PERSIST_DIR, "domains", safe)
        domain_collection = f"jarvis_domain_{safe}"
        # A first mail build takes hours, so never start one as a side effect of
        # launching; once the checkpoint exists later startups reload it warm.
        if (doc_set.type == "mail"
                and not os.path.exists(os.path.join(domain_persist, "owned_index.sqlite3"))
                and os.environ.get("JARVIS_BUILD_MAIL_INDEX", "0") != "1"):
            print(
                f"[jarvis] Domain corpus '{doc_set.name}': not indexed yet; skipping so startup stays fast. "
                "Re-launch once with JARVIS_BUILD_MAIL_INDEX=1 to build it (long first run, resumable)."
            )
            continue
        try:
            docs = _load_corpus_documents(doc_set.corpus_dir)
        except Exception as exc:
            # Class only: the exception text can embed the document that failed.
            print(f"[jarvis] Domain corpus '{doc_set.name}': load failed ({type(exc).__name__}); skipping.")
            continue
        if not docs:
            continue
        domain_index = None
        try:
            domain_index, _sc, _rm, _bk = _build_index(
                docs, embed_model, persist_dir=domain_persist, collection_name=domain_collection,
                **({"owned_corpus_dir": doc_set.corpus_dir} if doc_set.type == "mail" else {}),
            )
        except Exception as exc:
            print(f"[jarvis] Domain corpus '{doc_set.name}': index build failed ({type(exc).__name__}); skipping.")
        if domain_index is None:
            continue
        retriever = build_hybrid_retriever(
            domain_index, dense_top_k=top_k, bm25_top_k=top_k, fusion_top_k=top_k, llm=llm,
        )
        domains.append({
            "name": doc_set.name,
            "retriever": retriever,
            "index": domain_index,
            "match_any": doc_set.match_any,
            "label": doc_set.label or doc_set.name,
            "corpus_dir": os.path.abspath(doc_set.corpus_dir),
        })
        print(f"[jarvis] Domain corpus '{doc_set.name}' loaded ({len(docs)} docs); routes on {doc_set.match_any}.")
    return domains


def build_runtime(input_dir: str) -> QueryRuntime:
    """Perform all one-time startup work and return a ready `QueryRuntime`."""
    with Spinner("Loading language model") as spin:
        llm = _build_llm()
        spin.update("Loading embedding model")
        embed_model = _build_embed_model()

    if not os.environ.get("HF_TOKEN"):
        print(
            "Warning: HF_TOKEN is not set. HuggingFaceEmbedding may use anonymous access and be rate-limited.\n"
            "Set HF_TOKEN in your environment for authenticated downloads and better stability."
        )

    Settings.llm = llm
    Settings.embed_model = embed_model

    corpus_docs = _load_corpus_documents(input_dir)
    all_docs = corpus_docs or []

    img_docs, _img_chat_engine = _load_image_documents(input_dir, llm)
    if img_docs:
        all_docs.extend(img_docs)

    doc_chat_engine = None
    doc_query_engine = None
    doc_retriever = None
    doc_index = None
    storage_context = None
    response_mode = "query"

    if all_docs:
        doc_index, storage_context, response_mode, active_vector_backend = _build_index(all_docs, embed_model)

        doc_chat_engine = doc_index.as_chat_engine(chat_mode="condense_plus_context", llm=llm)
        retrieval_top_k = os.environ.get("JARVIS_RETRIEVAL_TOP_K", DEFAULT_RETRIEVAL_TOP_K)
        doc_retriever = build_hybrid_retriever(
            doc_index,
            dense_top_k=int(retrieval_top_k),
            bm25_top_k=int(os.environ.get("JARVIS_BM25_TOP_K", retrieval_top_k)),
            fusion_top_k=int(os.environ.get("JARVIS_FUSION_TOP_K", retrieval_top_k)),
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
        response_mode = "query"

    tts_manager = _build_tts()

    home_indexed_files, home_excluded_files = _partition_home_files(input_dir, all_docs)
    if home_excluded_files:
        print(
            "[jarvis] Launch-folder files present but not indexed "
            f"(no extractable data): {home_excluded_files}"
        )

    mcp_registry = load_mcp_registry()
    domain_retrievers = _build_domain_retrievers(embed_model, llm)

    # The speech worker loads on a background thread so it overlaps with the work
    # above; join it here so its readiness notice lands before the REPL prompt
    # rather than printing over it.
    if hasattr(tts_manager, "wait_until_ready"):
        tts_manager.wait_until_ready()

    return QueryRuntime(
        all_docs=all_docs,
        llm=llm,
        doc_retriever=doc_retriever,
        doc_chat_engine=doc_chat_engine,
        doc_index=doc_index,
        storage_context=storage_context,
        response_mode=response_mode,
        tts=tts_manager,
        persist_dir=PERSIST_DIR,
        home_corpus_dir=os.path.abspath(input_dir),
        home_indexed_files=home_indexed_files,
        home_excluded_files=home_excluded_files,
        mcp_registry=mcp_registry,
        domain_retrievers=domain_retrievers,
    )
