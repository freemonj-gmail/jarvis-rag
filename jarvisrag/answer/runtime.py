"""The runtime objects one Jarvis session shares across questions.

Kept in its own module so both the startup code that builds it
(``answer/bootstrap.py``) and the pipeline that uses it (``answer/pipeline.py``)
can import it without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass

from jarvisrag.config import STORAGE_DIR


@dataclass
class QueryRuntime:
    """Bundle of the initialized runtime objects `handle_query` needs.

    The main application (`Jarvis.py`) builds the index/retriever/LLM/TTS once at
    startup and passes them in here, keeping the entry script thin while the full
    RAG + fallback pipeline lives in the pipeline module."""
    all_docs: list
    llm: object
    doc_retriever: object | None
    doc_chat_engine: object | None = None
    doc_index: object | None = None
    storage_context: object | None = None
    response_mode: str = "query"
    tts: object | None = None
    persist_dir: str = STORAGE_DIR
    home_corpus_dir: str | None = None
    home_indexed_files: list[str] | None = None
    home_excluded_files: list[str] | None = None
    mcp_registry: object | None = None
    domain_retrievers: list | None = None
