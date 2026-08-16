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

    def retrieve(query: str) -> list[str]:
        with _private_runtime_output(args.verbose):
            evidence, _ = _retrieve_query_supported_evidence(
                query,
                runtime.doc_retriever,
                home_corpus_dir=runtime.home_corpus_dir,
            )
        return evidence

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
    report = evaluate_retrieval(
        cases,
        retrieve,
        include_case_details=args.case_details,
        metadata=metadata,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Retrieval evaluation report written ({report['case_count']} aggregate cases).")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
