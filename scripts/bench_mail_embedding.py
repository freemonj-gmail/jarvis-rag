"""Measure locally cached mail embeddings or a disposable checkpointed pipeline.

The default embedding comparison is read-only. The pipeline mode writes only to
private sample copies and a disposable localhost Milvus collection. Neither mode
opens the live Mail store, answer runtime, or connectors. Output is aggregate only.
"""

from __future__ import annotations

import argparse
import contextlib
import email
import hashlib
import json
import os
import random
import resource
import shutil
import statistics
import sys
import tempfile
import time
import uuid
from collections import Counter
from email import policy
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["JARVIS_DEBUG"] = "0"

from scripts.evaluate_retrieval import _private_runtime_output


class AlignedTokenizer:
    def __init__(self, tokenizer, multiple):
        self.tokenizer = tokenizer
        self.multiple = multiple

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **dict(kwargs, pad_to_multiple_of=self.multiple))

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def load_sample(corpus_dir: Path, cache_dir: Path, count: int, seed: int):
    from jarvisrag.extraction.extractors import (
        EXTRACTION_CACHE_VERSION,
        _deserialize_documents_from_cache,
        _extraction_cache_path,
        _extraction_mode_signature,
    )

    sources = sorted(corpus_dir.glob("*.emlx"))
    if count < 1 or len(sources) < count:
        raise ValueError("Sample size must be positive and no larger than the staged corpus.")
    selected = sorted(random.Random(seed).sample(sources, count))
    mode = _extraction_mode_signature()
    documents = []
    for source in selected:
        with open(_extraction_cache_path(str(cache_dir), str(source)), encoding="utf-8") as handle:
            cached = json.load(handle)
        source_stat = source.stat()
        if (
            cached.get("version") != EXTRACTION_CACHE_VERSION
            or cached.get("extraction_mode") != mode
            or cached.get("file_path") != str(source.resolve())
            or cached.get("mtime_ns") != source_stat.st_mtime_ns
            or cached.get("size") != source_stat.st_size
        ):
            raise ValueError("Sample contains a stale cache entry; no extraction was attempted.")
        documents.extend(_deserialize_documents_from_cache(cached.get("documents") or []))
    return documents, len(sources)


def prepare_batches(documents, docs_per_batch: int):
    from llama_index.core import Settings
    from llama_index.core.ingestion import run_transformations
    from jarvisrag.extraction.indexing import (
        chunk_documents,
        ensure_document_metadata,
        filter_informative_documents,
        split_large_docs,
    )

    documents = filter_informative_documents(documents)
    for document in documents:
        ensure_document_metadata(document)
    prepared = filter_informative_documents(split_large_docs(documents))
    batches = [
        run_transformations(batch, Settings.transformations)
        for batch in chunk_documents(prepared, docs_per_chunk=docs_per_batch)
    ]
    return batches, len(prepared)


@contextlib.contextmanager
def pilot_directory(parent: Path):
    """Keep metadata paths repeatable and refuse concurrent or leftover pilot state."""
    directory = parent / f"jarvis_mail_pilot_{os.getuid()}"
    directory.mkdir(mode=0o700)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def copy_sample(documents, corpus_dir: Path, destination: Path):
    """Copy staged sources into a private pilot corpus without modifying the originals."""
    root = corpus_dir.resolve()
    destination.mkdir(mode=0o700)
    copied = {}
    result = []
    for document in documents:
        source = Path(document.metadata["file_path"]).resolve()
        if not source.is_relative_to(root) or source.suffix.lower() != ".emlx":
            raise ValueError("Pilot source is outside the staged mail corpus.")
        if source not in copied:
            target = destination / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            before = source.stat()
            shutil.copyfile(source, target)
            os.chmod(target, 0o600)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or target.stat().st_size != before.st_size:
                raise ValueError("Staged source changed during pilot copying; stop concurrent refreshes.")
            copied[source] = target
        clone = document.model_copy(deep=True)
        clone.metadata["file_path"] = str(copied[source])
        result.append(clone)
    return result


@contextlib.contextmanager
def measure_pipeline(embed_model):
    """Observe real calls without replacing their computations or return values."""
    from llama_index.core.storage.storage_context import StorageContext
    from jarvisrag.extraction import indexing, owned_index
    from jarvisrag.vectorstore import MilvusVectorStore

    metrics = {"seconds": {}, "calls": {}, "embedded_texts": 0, "prepared_sources": 0, "inserted_vectors": 0}

    def observe(name, function):
        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                if name == "embedding":
                    if args[0] is not embed_model:
                        raise RuntimeError("Unexpected embedding model during pilot measurement.")
                    result = function(*args[1:], **kwargs)
                else:
                    result = function(*args, **kwargs)
                if name == "embedding":
                    metrics["embedded_texts"] += len(args[1])
                elif name == "preparation":
                    metrics["prepared_sources"] += result
                elif name == "vector_write":
                    metrics["inserted_vectors"] += len(args[1])
                return result
            finally:
                metrics["seconds"][name] = metrics["seconds"].get(name, 0.0) + time.perf_counter() - started
                metrics["calls"][name] = metrics["calls"].get(name, 0) + 1
        return measured

    targets = [
        (owned_index, "_prepare_cache", "preparation"),
        (type(embed_model), "get_text_embedding_batch", "embedding"),
        (MilvusVectorStore, "add", "vector_write"),
        (indexing, "_flush_vector_store", "flush"),
        (MilvusVectorStore, "indexed_node_ids", "vector_verification"),
        (MilvusVectorStore, "delete_nodes", "vector_delete"),
        (StorageContext, "persist", "persistence"),
    ]
    started = time.perf_counter()
    with contextlib.ExitStack() as stack:
        for target, attribute, name in targets:
            original = getattr(embed_model, attribute) if name == "embedding" else getattr(target, attribute)
            stack.enter_context(patch.object(target, attribute, new=observe(name, original)))
        try:
            yield metrics
        finally:
            metrics["wall_seconds"] = time.perf_counter() - started


def revise_sample_message(path: Path, corpus_dir: Path):
    """Revise only a copied header while preserving the original MIME body and plist."""
    from jarvisrag.extraction.extractors import EmlxReader

    if not path.resolve().is_relative_to(corpus_dir.resolve()):
        raise ValueError("Refusing to revise a source outside the private pilot corpus.")
    raw = path.read_bytes()
    message_bytes = EmlxReader._message_bytes(raw)
    offset = raw.find(b"\n") + 1
    if not offset or int(raw[:offset].strip()) != len(message_bytes):
        raise ValueError("Pilot revision requires valid emlx framing.")
    message = email.message_from_bytes(message_bytes, policy=policy.default)
    before_body = EmlxReader._body_text(message)
    subject = str(message.get("Subject", "")) + " [pilot revision]"
    if "Subject" in message:
        message.replace_header("Subject", subject)
    else:
        message["Subject"] = subject
    payload = message.as_bytes(policy=policy.default)
    reparsed = email.message_from_bytes(payload, policy=policy.default)
    if EmlxReader._body_text(reparsed) != before_body:
        raise ValueError("Header revision unexpectedly changed the extracted body.")
    path.write_bytes(str(len(payload)).encode("ascii") + b"\n" + payload + raw[offset + len(message_bytes):])
    return EmlxReader().load_data(str(path))


def pipeline_pilot(args):
    import numpy as np
    import torch
    from llama_index.core.schema import MetadataMode
    from jarvisrag.answer.bootstrap import EXTRACTION_CACHE_DIR, _build_embed_model
    from jarvisrag.extraction.indexing import build_or_load_persistent_index
    from jarvisrag.iotools import _is_port_open
    from jarvisrag.vectorstore import MilvusVectorStore, create_milvus_vector_store

    if not _is_port_open("127.0.0.1", "19530", timeout=2.0):
        raise RuntimeError("Pipeline pilot requires local Milvus on 127.0.0.1:19530.")
    started = time.perf_counter()
    with _private_runtime_output(False):
        documents, corpus_count = load_sample(
            args.corpus_dir.resolve(), args.cache_dir or Path(EXTRACTION_CACHE_DIR), args.sample, args.seed,
        )
    report = {"sample_messages": args.sample, "corpus_messages": corpus_count, "cache_load_seconds": time.perf_counter() - started}
    print("Loading cached BGE-M3 for the isolated pipeline pilot...", flush=True)
    started = time.perf_counter()
    with _private_runtime_output(False):
        embed_model = _build_embed_model()
    report["model_load_seconds"] = time.perf_counter() - started
    report["device"] = embed_model._device
    report["embedding_batch_size"] = embed_model.embed_batch_size
    report["embedding_dtype"] = str(next(embed_model._model.parameters()).dtype)
    collection = "jarvis_mail_pilot_" + uuid.uuid4().hex
    store = create_milvus_vector_store(collection_name=collection, host="127.0.0.1", port="19530")
    with _private_runtime_output(False):
        if store.has_collection():
            raise RuntimeError("Refusing to reuse an existing pilot collection.")
    try:
        with pilot_directory(Path(tempfile.gettempdir())) as directory:
            root = Path(directory, "corpus")
            persist = str(Path(directory, "index"))
            started = time.perf_counter()
            with _private_runtime_output(False):
                copied = copy_sample(documents, args.corpus_dir, root)
                batches, prepared_count = prepare_batches([document.model_copy(deep=True) for document in copied], 64)
                baseline = [node for batch in batches for node in batch]
                lengths = [len(tokens) for tokens in embed_model._model.tokenizer(
                    [node.get_content(MetadataMode.EMBED) for node in baseline], truncation=False, padding=False,
                )["input_ids"]]
            if not lengths or max(lengths) > embed_model.max_length:
                raise ValueError("Pilot has no valid embedding inputs or would truncate content.")
            report["sample_copy_and_reference_preparation_seconds"] = time.perf_counter() - started
            report["prepared_documents"] = prepared_count
            report["actual_nodes"] = len(baseline)
            report["embedding_input_sha256"] = hashlib.sha256(json.dumps([
                (node.get_content(MetadataMode.EMBED), node.metadata) for node in baseline
            ], sort_keys=True).encode("utf-8")).hexdigest()
            report["bge_tokens_p50_p95_max"] = np.percentile(lengths, [50, 95, 100]).tolist()
            report["stages"] = {}

            def build_stage(name, source_documents, interrupt=False):
                print(f"Pipeline pilot: {name}...", flush=True)
                if embed_model._device == "mps":
                    torch.mps.synchronize()
                    if name == "cold_build":
                        torch.mps.empty_cache()
                with _private_runtime_output(False), measure_pipeline(embed_model) as metrics:
                    write = MilvusVectorStore.add

                    def partial_write(active_store, nodes, **kwargs):
                        write(active_store, nodes[:max(1, len(nodes) // 2)], **kwargs)
                        raise InterruptedError("Pilot interruption after a partial vector write.")

                    with patch.object(MilvusVectorStore, "add", new=partial_write) if interrupt else contextlib.nullcontext():
                        try:
                            result = build_or_load_persistent_index(
                                source_documents, store, persist, embed_model,
                                owned_corpus_dir=str(root), owned_extensions=(".emlx",),
                            )
                        except InterruptedError:
                            if not interrupt:
                                raise
                            result = None
                        else:
                            if interrupt:
                                raise AssertionError("The pilot interruption was not exercised.")
                if embed_model._device == "mps":
                    torch.mps.synchronize()
                    metrics["mps_driver_gib_after_stage_not_peak"] = torch.mps.driver_allocated_memory() / 1024**3
                report["stages"][name] = metrics
                print(f"Completed {name}: {metrics['wall_seconds']:.2f}s, {metrics['embedded_texts']} embedded texts, {metrics['inserted_vectors']} vector writes", flush=True)
                return result

            def require_parity(index, reference):
                expected = Counter((node.get_content(MetadataMode.EMBED), json.dumps(node.metadata, sort_keys=True)) for node in reference)
                actual = Counter((node.get_content(MetadataMode.EMBED), json.dumps(node.metadata, sort_keys=True)) for node in index.docstore.docs.values())
                if actual != expected or store.indexed_node_ids() != set(index.docstore.docs):
                    raise AssertionError("Pilot node content, metadata or vector completeness differs from the reference.")

            index, _ = build_stage("cold_build", copied)
            with _private_runtime_output(False):
                require_parity(index, baseline)
            report["exact_embedding_input_and_metadata_parity"] = True
            store._client.close()
            store = create_milvus_vector_store(collection_name=collection, host="127.0.0.1", port="19530")
            index, _ = build_stage("warm_restart", copied)
            warm = report["stages"]["warm_restart"]
            if warm["embedded_texts"] or warm["prepared_sources"] or warm["inserted_vectors"]:
                raise AssertionError("Warm startup unexpectedly repeated source preparation or vector work.")
            source_counts = Counter(node.metadata["file_path"] for node in index.docstore.docs.values())
            candidates = sorted((count, path) for path, count in source_counts.items() if count > 1)
            if not candidates:
                raise ValueError("Pilot needs a multi-chunk message to verify a partial-write recovery.")
            selected_path = Path(candidates[0][1])
            with _private_runtime_output(False):
                updated = revise_sample_message(selected_path, root)
                if not updated:
                    raise ValueError("Copied revision did not produce an extracted document.")
                originals = [document for document in copied if document.metadata["file_path"] == str(selected_path)]
                if len(originals) != 1 or len(updated) != 1:
                    raise ValueError("Pilot revision currently requires one extracted document per message.")
                merged = dict(originals[0].metadata)
                merged.update(updated[0].metadata)
                updated[0].metadata = merged
                changed = [document for document in copied if document.metadata["file_path"] != str(selected_path)] + updated
            build_stage("interrupted_update", changed, interrupt=True)
            store._client.close()
            store = create_milvus_vector_store(collection_name=collection, host="127.0.0.1", port="19530")
            index, _ = build_stage("resume_update", changed)
            if report["stages"]["resume_update"]["embedded_texts"]:
                raise AssertionError("Interrupted update did not reuse all checkpointed embeddings.")
            if not report["stages"]["resume_update"]["inserted_vectors"]:
                raise AssertionError("Pilot recovery did not complete a partial vector write.")
            with _private_runtime_output(False):
                changed_batches, _ = prepare_batches([document.model_copy(deep=True) for document in changed], 64)
                require_parity(index, [node for batch in changed_batches for node in batch])
                selected_path.unlink()
                remaining = [document for document in changed if document.metadata["file_path"] != str(selected_path)]
            index, _ = build_stage("source_deletion", remaining)
            if report["stages"]["source_deletion"]["embedded_texts"]:
                raise AssertionError("Source deletion unexpectedly re-embedded unchanged messages.")
            with _private_runtime_output(False):
                remaining_batches, _ = prepare_batches([document.model_copy(deep=True) for document in remaining], 64)
                require_parity(index, [node for batch in remaining_batches for node in batch])
            report["source_change_resume_and_deletion_verified"] = True
            report["checkpoint_bytes_after_delta"] = sum(path.stat().st_size for path in Path(persist).rglob("*") if path.is_file())
            report["cold_sample_scaled_hours_not_full_build_eta"] = report["stages"]["cold_build"]["wall_seconds"] * corpus_count / args.sample / 3600
            report["scope"] = "Copied text-only sample with real BGE-M3 and local Milvus. Warm reload and injected partial-write recovery reconnect the client, not the process/server. Excludes full-corpus inventory/summary costs, BM25, answer accuracy, attachments/images and concurrent-load variability. Scaling is not a promised build time."
    finally:
        with _private_runtime_output(False):
            if store._client is not None:
                try:
                    if store._client.has_collection(collection):
                        store._client.drop_collection(collection)
                    if store._client.has_collection(collection):
                        raise RuntimeError("Disposable pilot collection cleanup failed.")
                finally:
                    store._client.close()
        print("Disposable pilot collection and private sample directory cleaned up.", flush=True)
    print(json.dumps(report, indent=2))
    return 0


def embedding_order(lengths: list[int], batch_sizes: list[int], sort: bool):
    if sum(batch_sizes) != len(lengths):
        raise ValueError("Batch boundaries must cover every node exactly once.")
    groups = []
    start = 0
    for size in batch_sizes:
        indices = list(range(start, start + size))
        if sort:
            indices.sort(key=lambda index: lengths[index])
        groups.append(indices)
        start += size
    return groups


def padded_tokens(groups, lengths, batch_size):
    return sum(
        max(lengths[index] for index in group[start:start + batch_size])
        * len(group[start:start + batch_size])
        for group in groups
        for start in range(0, len(group), batch_size)
    )


def measure(embed_model, texts, groups, release_cache=False):
    import numpy as np
    import torch

    def synchronize():
        if embed_model._device == "mps":
            torch.mps.synchronize()
        elif embed_model._device.startswith("cuda"):
            torch.cuda.synchronize()

    vectors = [None] * len(texts)
    allocations = []
    synchronize()
    started = time.perf_counter()
    for group in groups:
        embedded = embed_model.get_text_embedding_batch([texts[index] for index in group])
        if len(embedded) != len(group):
            raise ValueError("Embedding response did not cover the input batch.")
        for index, vector in zip(group, embedded):
            vectors[index] = vector
        if embed_model._device == "mps":
            synchronize()
            before = torch.mps.driver_allocated_memory()
            if release_cache:
                torch.mps.empty_cache()
            allocations.append({
                "tensor_gib": torch.mps.current_allocated_memory() / 1024**3,
                "driver_before_gib": before / 1024**3,
                "driver_after_gib": torch.mps.driver_allocated_memory() / 1024**3,
            })
    synchronize()
    elapsed = time.perf_counter() - started
    result = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite embeddings detected.")
    return elapsed, result, allocations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("embedding", "pipeline"), default="embedding")
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "jarvis_storage/mail_corpus")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--sample", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--candidate-batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--compare-cache-release", action="store_true")
    parser.add_argument("--compare-padding", type=int, default=0)
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if any(size < 1 or size > 32 for size in args.candidate_batch_sizes):
        parser.error("Candidate batch sizes must be between 1 and the current default of 32.")
    if args.compare_padding < 0 or (args.compare_padding and args.compare_cache_release):
        parser.error("Choose one comparison; padding multiple must be positive.")
    if args.mode == "pipeline":
        if args.compare_padding or args.compare_cache_release:
            parser.error("Pipeline mode preserves the current embedding configuration.")
        return pipeline_pilot(args)

    import numpy as np
    import torch
    from llama_index.core.schema import MetadataMode
    from jarvisrag.answer.bootstrap import EXTRACTION_CACHE_DIR, _build_embed_model

    print("Reading cached sample and applying existing transformations...", flush=True)
    started = time.perf_counter()
    with _private_runtime_output(False):
        documents, corpus_count = load_sample(
            args.corpus_dir.resolve(), args.cache_dir or Path(EXTRACTION_CACHE_DIR), args.sample, args.seed,
        )
        batches, prepared_count = prepare_batches(
            documents, max(1, int(os.environ.get("JARVIS_INDEX_BATCH_SIZE", "64"))),
        )
        texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for batch in batches for node in batch]
    prepare_seconds = time.perf_counter() - started
    if not texts:
        raise ValueError("No embedding inputs survived existing preparation.")

    print("Loading cached embedding model (offline)...", flush=True)
    with _private_runtime_output(False):
        embed_model = _build_embed_model()
        if args.compare_padding and embed_model.max_length % args.compare_padding:
            raise ValueError("Padding multiple must divide the model token limit.")
        lengths = [len(tokens) for tokens in embed_model._model.tokenizer(
            texts, truncation=False, padding=False,
        )["input_ids"]]
        if max(lengths) > embed_model.max_length:
            raise ValueError("Existing inputs exceed the model token limit; refusing a truncated benchmark.")
        embed_model.get_text_embedding_batch(texts[:embed_model.embed_batch_size])

    baseline_batch_size = embed_model.embed_batch_size
    batch_sizes = [len(batch) for batch in batches]
    schedules = {"baseline": embedding_order(lengths, batch_sizes, False)}
    embedding_batch_sizes = {"baseline": baseline_batch_size}
    if args.compare_cache_release:
        if embed_model._device != "mps":
            raise ValueError("Cache release comparison requires MPS.")
        schedules["release_cache"] = schedules["baseline"]
        embedding_batch_sizes["release_cache"] = baseline_batch_size
    if args.compare_padding:
        schedules["aligned_padding"] = schedules["baseline"]
        embedding_batch_sizes["aligned_padding"] = baseline_batch_size
    for size in ([] if args.compare_cache_release or args.compare_padding else sorted(set(args.candidate_batch_sizes))):
        name = f"bucketed_{size}"
        schedules[name] = embedding_order(lengths, batch_sizes, True)
        embedding_batch_sizes[name] = size
    timings = {name: [] for name in schedules}
    observed_mps_driver = {name: [] for name in schedules}
    allocation_traces = {name: [] for name in schedules}
    reference = None
    comparisons = {name: {"min_cosine": 1.0, "max_absolute_difference": 0.0} for name in schedules}
    for repeat in range(args.repeat):
        names = list(schedules) if repeat % 2 == 0 else list(reversed(schedules))
        for name in names:
            embed_model.embed_batch_size = embedding_batch_sizes[name]
            if embed_model._device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            print(f"Embedding {len(texts)} identical nodes: {name}, trial {repeat + 1}...", flush=True)
            transformer = embed_model._model[0]
            tokenizer_context = (
                patch.object(transformer, "tokenizer", AlignedTokenizer(transformer.tokenizer, args.compare_padding))
                if name == "aligned_padding" else contextlib.nullcontext()
            )
            with _private_runtime_output(False), tokenizer_context:
                elapsed, vectors, allocations = measure(
                    embed_model, texts, schedules[name], release_cache=name == "release_cache",
                )
            timings[name].append(elapsed)
            allocation_traces[name].append(allocations)
            if embed_model._device == "mps":
                observed_mps_driver[name].append(torch.mps.driver_allocated_memory() / 1024**3)
            if reference is None:
                reference = vectors
            else:
                cosine = np.sum(reference * vectors, axis=1) / (
                    np.linalg.norm(reference, axis=1) * np.linalg.norm(vectors, axis=1)
                )
                comparisons[name]["min_cosine"] = min(comparisons[name]["min_cosine"], float(cosine.min()))
                comparisons[name]["max_absolute_difference"] = max(
                    comparisons[name]["max_absolute_difference"], float(np.abs(reference - vectors).max()),
                )
            print(f"Completed {name}: {elapsed:.2f}s ({len(texts) / elapsed:.2f} nodes/s)", flush=True)

    medians = {name: statistics.median(values) for name, values in timings.items()}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "sample_messages": args.sample,
        "corpus_messages": corpus_count,
        "prepared_documents": prepared_count,
        "actual_nodes": len(texts),
        "prepare_seconds": prepare_seconds,
        "device": embed_model._device,
        "torch_threads": torch.get_num_threads(),
        "bge_tokens_p50_p95_max": np.percentile(lengths, [50, 95, 100]).tolist(),
        "padded_tokens": {
            name: padded_tokens(
                groups,
                [((length + args.compare_padding - 1) // args.compare_padding) * args.compare_padding for length in lengths]
                if name == "aligned_padding" else lengths,
                embedding_batch_sizes[name],
            ) for name, groups in schedules.items()
        },
        "trial_seconds": timings,
        "speedup": {name: medians["baseline"] / seconds for name, seconds in medians.items()},
        "embedding_only_projected_hours": {
            name: seconds * corpus_count / args.sample / 3600 for name, seconds in medians.items()
        },
        "vector_comparison_to_baseline": comparisons,
        "mps_driver_gib_after_trials_not_peak": observed_mps_driver,
        "mps_allocations_at_source_batch_boundaries_not_peak": allocation_traces,
        "peak_process_rss_gib": rss * (1 if sys.platform == "darwin" else 1024) / 1024**3,
        "scope": "Sample estimate only; excludes full preparation, model load, insertion, persistence, BM25, and answer accuracy.",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Benchmark stopped ({type(exc).__name__}); private exception details suppressed.", file=sys.stderr)
        raise SystemExit(1) from None