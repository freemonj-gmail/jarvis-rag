"""The runtime objects one Jarvis session shares across questions.

Kept in its own module so both the startup code that builds it
(``answer/bootstrap.py``) and the pipeline that uses it (``answer/pipeline.py``)
can import it without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jarvisrag.config import STORAGE_DIR


@dataclass
class Domain:
    """One domain corpus kept separate from the main index (e.g. federal
    benefits, personal mail), consulted when a question routes to it or as a
    cascade when the main sources do not answer."""
    name: str
    retriever: object = None
    index: object = None
    match_any: list[str] = field(default_factory=list)
    label: str = ""
    corpus_dir: str = ""


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
    domains: list[Domain] | None = None
