"""Run a retrieval-only Jarvis evaluation against local ground truth.

Keep private case files and reports under ``jarvis_storage/evaluation/`` so
queries, filenames, and expected passages remain gitignored.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("JARVIS_ENABLE_TTS", "0")

from jarvisrag.evaluation import evaluate_retrieval, load_retrieval_cases


def _directory_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for file_name in files:
            try:
                total += os.path.getsize(os.path.join(root, file_name))
            except OSError:
                continue
    return total


def _manifest_chunk_count(persist_dir: str) -> int | None:
    manifest_path = os.path.join(persist_dir, "jarvis_index_manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    documents = manifest.get("documents") if isinstance(manifest, dict) else None
    return len(documents) if isinstance(documents, list) else None


def _docstore_node_count(runtime: Any) -> int | None:
    storage_context = getattr(runtime, "storage_context", None)
    docstore = getattr(storage_context, "docstore", None)
    docs = getattr(docstore, "docs", None)
    try:
        return len(docs) if docs is not None else None
    except TypeError:
        return None


def _vector_count(runtime: Any) -> int | None:
    storage_context = getattr(runtime, "storage_context", None)
    vector_stores = getattr(storage_context, "vector_stores", {}) or {}
    counts: list[int] = []
    for vector_store in vector_stores.values():
        get_size = getattr(vector_store, "_get_collection_size", None)
        if callable(get_size):
            try:
                counts.append(int(get_size()))
                continue
            except Exception:
                pass
        data = getattr(vector_store, "data", None)
        embeddings = getattr(data, "embedding_dict", None)
        if embeddings is not None:
            counts.append(len(embeddings))
    return sum(counts) if counts else None


RETRIEVER_MODES = ("hybrid", "dense", "bm25")


def _mode_retrievers(retriever: Any, modes: list[str]) -> dict[str, Any]:
    """Split the runtime's hybrid retriever into its own BM25 and dense parts.

    Every mode is the SAME object the runtime built (same index, same top-k), so
    a comparison isolates the retrieval method and nothing else.
    """

    from llama_index.retrievers.bm25 import BM25Retriever

    parts = list(getattr(retriever, "_retrievers", None) or [])
    available: dict[str, Any] = {"hybrid": retriever if parts else None}
    available["bm25"] = next((part for part in parts if isinstance(part, BM25Retriever)), None)
    available["dense"] = next((part for part in parts if not isinstance(part, BM25Retriever)), None)
    if not parts:
        available["dense"] = retriever
    docstore = getattr(retriever, "_jarvis_docstore", None)
    for mode in ("bm25", "dense"):
        part = available[mode]
        if part is not None and docstore is not None and getattr(part, "_jarvis_docstore", None) is None:
            with contextlib.suppress(Exception):
                part._jarvis_docstore = docstore
    missing = [mode for mode in modes if available.get(mode) is None]
    if missing:
        raise RuntimeError(
            f"retriever mode(s) {', '.join(missing)} unavailable: the runtime did not build a hybrid retriever"
        )
    return {mode: available[mode] for mode in modes}


def _parse_modes(value: str) -> list[str]:
    modes = [mode.strip().lower() for mode in value.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in RETRIEVER_MODES]
    if not modes or unknown:
        raise argparse.ArgumentTypeError(
            f"choose one or more of {', '.join(RETRIEVER_MODES)} (comma-separated)"
        )
    return list(dict.fromkeys(modes))


@contextlib.contextmanager
def _private_runtime_output(verbose: bool):
    """Hide bootstrap/retrieval logs that can contain private document text."""

    if verbose:
        yield
        return
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with open(os.devnull, "w", encoding="utf-8") as null_stream:
            with contextlib.redirect_stdout(null_stream), contextlib.redirect_stderr(null_stream):
                yield
    finally:
        logging.disable(previous_disable)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Jarvis retrieval without answer synthesis or MCP calls. "
            "The report omits queries, filenames, and evidence text."
        )
    )
    parser.add_argument("corpus", help="local corpus directory used to build the runtime")
    parser.add_argument("cases", help="local schema_version=1 retrieval case JSON file")
    parser.add_argument("--output", help="optional JSON report path")
    parser.add_argument(
        "--persist-dir",
        help="isolated local metadata/cache directory (sets JARVIS_PERSIST_DIR)",
    )
    parser.add_argument(
        "--case-details",
        action="store_true",
        help="include case IDs, ranks, counts, and latency; never includes query/evidence text",
    )
    parser.add_argument(
        "--retrievers",
        type=_parse_modes,
        default=["hybrid"],
        help=(
            "comma-separated retrieval methods to score from one runtime build: "
            "hybrid (production), dense, bm25; one report per method"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show bootstrap/retrieval logs; may expose private filenames and extracted text",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = load_retrieval_cases(args.cases)
    if args.persist_dir:
        os.environ["JARVIS_PERSIST_DIR"] = os.path.abspath(os.path.expanduser(args.persist_dir))

    from jarvisrag.answer.bootstrap import DEFAULT_RETRIEVAL_TOP_K, build_runtime
    from jarvisrag.retrieval import _retrieve_query_supported_evidence

    print("Building isolated retrieval runtime...", flush=True)
    startup_started = time.perf_counter()
    with _private_runtime_output(args.verbose):
        runtime = build_runtime(args.corpus)
    startup_seconds = time.perf_counter() - startup_started
    if runtime.doc_retriever is None:
        raise RuntimeError("the corpus produced no retriever")
    retrievers = _mode_retrievers(runtime.doc_retriever, args.retrievers)

    def retriever_for(mode_retriever: Any):
        def retrieve(query: str) -> list[str]:
            with _private_runtime_output(args.verbose):
                evidence, _ = _retrieve_query_supported_evidence(
                    query,
                    mode_retriever,
                    home_corpus_dir=runtime.home_corpus_dir,
                )
            return evidence

        return retrieve

    manifest_chunk_count = _manifest_chunk_count(runtime.persist_dir)
    vector_count = _vector_count(runtime)
    metadata: dict[str, Any] = {
        "startup_seconds": startup_seconds,
        "manifest_chunk_count": manifest_chunk_count,
        "docstore_node_count": _docstore_node_count(runtime),
        "vector_count": vector_count,
        "vector_expansion_ratio": (
            vector_count / manifest_chunk_count
            if vector_count is not None and manifest_chunk_count
            else None
        ),
        "persistence_bytes": _directory_size(runtime.persist_dir),
        "vector_backend_requested": os.environ.get("JARVIS_VECTOR_BACKEND", "milvus"),
        "milvus_collection": os.environ.get("MILVUS_COLLECTION", "jarvis_context"),
        "retrieval_top_k": int(os.environ.get("JARVIS_RETRIEVAL_TOP_K", DEFAULT_RETRIEVAL_TOP_K)),
        "max_evidence_lines": int(os.environ.get("JARVIS_MAX_EVIDENCE_LINES", "30")),
    }
    reports = {
        mode: evaluate_retrieval(
            cases,
            retriever_for(mode_retriever),
            include_case_details=args.case_details,
            metadata={**metadata, "retriever_mode": mode},
        )
        for mode, mode_retriever in retrievers.items()
    }
    report = reports[args.retrievers[0]] if len(reports) == 1 else {"by_retriever": reports}
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Retrieval evaluation report written ({len(cases)} aggregate cases, {len(reports)} method(s)).")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
