"""Checkpoint support for an explicitly owned corpus and exclusive vector store.

General cross-directory indexing continues to use the existing manifest path.
The private SQLite ledger stores prepared children and completed embeddings so
an interrupted owned-corpus build can reconcile every expected child on restart.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import resource
import sqlite3
import struct
import sys
import time
from collections import defaultdict
from contextlib import closing, contextmanager
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.ingestion import run_transformations
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.simple import SimpleVectorStore

from jarvisrag.iotools import Spinner
from jarvisrag.extraction import indexing


def _digest(value):
    def serialize(item):
        if callable(item):
            return f"{item.__module__}.{item.__qualname__}"
        raise TypeError(f"Unsupported checkpoint signature value: {type(item).__name__}")

    return hashlib.sha256(json.dumps(value, sort_keys=True, default=serialize).encode()).hexdigest()


@contextmanager
def _ledger(persist_dir, owner):
    os.makedirs(persist_dir, mode=0o700, exist_ok=True)
    os.chmod(persist_dir, 0o700)
    lock_path = os.path.join(persist_dir, "owned_index.lock")
    with open(lock_path, "a") as lock:
        os.chmod(lock_path, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("An owned-corpus build is already running.") from None
        database = os.path.join(persist_dir, "owned_index.sqlite3")
        with closing(sqlite3.connect(database)) as connection:
            os.chmod(database, 0o600)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            # The 2 MB default cache thrashes a multi-GB ledger once the WAL grows.
            connection.execute("PRAGMA cache_size=-131072")
            connection.execute("PRAGMA wal_autocheckpoint=2000")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sources (path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS children (
                    id TEXT PRIMARY KEY, path TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    payload TEXT NOT NULL, embedding BLOB
                );
                CREATE INDEX IF NOT EXISTS children_path ON children(path, ordinal);
            """)
            # Ledgers written before float32 storage have no dims column; a NULL
            # dims marks a legacy float64 blob so existing work stays readable.
            if "dims" not in {row[1] for row in connection.execute("PRAGMA table_info(children)")}:
                with connection:
                    connection.execute("ALTER TABLE children ADD COLUMN dims INTEGER")
            row = connection.execute("SELECT value FROM config WHERE key='owner'").fetchone()
            if row and row[0] != owner:
                raise ValueError("Checkpoint belongs to a different corpus or vector store.")
            with connection:
                connection.execute("INSERT OR IGNORE INTO config VALUES ('owner', ?)", (owner,))
            yield connection


def _source_groups(documents, corpus_dir, extensions):
    root = Path(corpus_dir).resolve()
    if not root.is_dir() or not extensions:
        raise ValueError("Owned indexing requires an existing corpus directory and explicit source extensions.")
    grouped = defaultdict(list)
    for document in documents:
        metadata = document.metadata or {}
        if metadata.get("is_file_list"):
            key = "__summary__"
        else:
            path = Path(str(metadata.get("file_path", ""))).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Owned corpus contains an absent or out-of-scope source.")
            key = str(path)
        grouped[key].append(document)

    def scan_error(error):
        raise OSError("Cannot verify the complete owned-corpus source inventory.") from error

    discovered = set()
    for folder, _, names in os.walk(root, onerror=scan_error):
        for name in names:
            if name.lower().endswith(tuple(extensions)):
                path = Path(folder, name).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Owned corpus contains an out-of-scope link.")
                discovered.add(str(path))
    if discovered != set(grouped) - {"__summary__"}:
        raise ValueError("Source inventory and extracted documents disagree; no deletions are safe.")
    return grouped


def _prepare_children(documents, signature):
    source_documents = indexing.filter_informative_documents([
        document.model_copy(deep=True) for document in documents
    ])
    for document in source_documents:
        indexing.ensure_document_metadata(document)
    prepared = indexing.filter_informative_documents(indexing.split_large_docs(source_documents))
    nodes = run_transformations(prepared, Settings.transformations)
    identities = {}
    for ordinal, node in enumerate(nodes):
        if not isinstance(node, TextNode):
            raise TypeError("Owned checkpoint currently supports text nodes only.")
        identities[node.node_id] = "owned-" + _digest([signature, node.ref_doc_id, ordinal, node.hash])
    for node in nodes:
        node.id_ = identities[node.node_id]
        for related in node.relationships.values():
            for relation in related if isinstance(related, list) else [related]:
                if relation.node_id in identities:
                    relation.node_id = identities[relation.node_id]
    return nodes


def _prepare_cache(connection, grouped, signature, rebuild=False):
    previous = dict(connection.execute("SELECT path, fingerprint FROM sources"))
    prepared_count = 0
    with Spinner(f"Checking prepared cache ({len(grouped)} sources)") as spinner:
        for path, documents in grouped.items():
            fingerprint = _digest([signature, [document.hash for document in documents]])
            if not rebuild and previous.get(path) == fingerprint:
                continue
            nodes = _prepare_children(documents, signature)
            with connection:
                connection.execute("DELETE FROM children WHERE path=?", (path,))
                connection.executemany(
                    "INSERT INTO children (id, path, ordinal, payload, embedding, dims)"
                    " VALUES (?, ?, ?, ?, NULL, NULL)",
                    [(node.node_id, path, ordinal, node.to_json()) for ordinal, node in enumerate(nodes)],
                )
                connection.execute("INSERT OR REPLACE INTO sources VALUES (?, ?)", (path, fingerprint))
            prepared_count += 1
            if prepared_count % 1000 == 0:
                spinner.update(f"Prepared {prepared_count} changed sources")
        with connection:
            for removed in previous.keys() - grouped.keys():
                connection.execute("DELETE FROM children WHERE path=?", (removed,))
                connection.execute("DELETE FROM sources WHERE path=?", (removed,))
    print(f"[index] Prepared {prepared_count} source(s); reused {len(grouped) - prepared_count} cached source(s).")
    return prepared_count


def _embedding_blob(vector):
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("Cannot checkpoint an empty or non-finite embedding.")
    return struct.pack(f"<{len(vector)}f", *vector)


def _embedding_values(blob, dims):
    """Read a checkpointed vector; a NULL ``dims`` means a legacy float64 blob."""
    if dims:
        return list(struct.unpack(f"<{dims}f", blob))
    return list(struct.unpack(f"<{len(blob) // 8}d", blob))


_BUCKET_BATCH = 8
_READ_PAGE = 512
_FLUSH_EVERY = 50_000
_PROGRESS_SECONDS = 60


def _resident_bytes():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage if sys.platform == "darwin" else usage * 1024


def _report_progress(installed, total, started, persist_dir):
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = installed / elapsed
    eta = f"{(total - installed) / rate / 3600:.1f}h" if rate else "unknown"
    ledger = os.path.join(persist_dir, "owned_index.sqlite3")
    sizes = []
    for suffix, label in (("", "db"), ("-wal", "wal")):
        try:
            sizes.append(f"{label} {os.path.getsize(ledger + suffix) / 1024 ** 3:.2f}G")
        except OSError:
            pass
    print(
        f"[index] {installed}/{total} child vectors ({installed / total:.1%}) · "
        f"{rate:.1f}/s · eta {eta} · rss {_resident_bytes() / 1024 ** 3:.2f}G · "
        + " · ".join(sizes),
        flush=True,
    )


def _embed_length_bucketed(embed_model, nodes):
    """Embed similar-length texts together so padding does not dominate a batch.

    Returns embeddings in the caller's original node order."""
    texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    previous = getattr(embed_model, "embed_batch_size", None)
    if previous is not None:
        embed_model.embed_batch_size = min(previous, _BUCKET_BATCH)
    try:
        packed = embed_model.get_text_embedding_batch([texts[index] for index in order])
    finally:
        if previous is not None:
            embed_model.embed_batch_size = previous
    if len(packed) != len(texts):
        return packed
    restored = [None] * len(texts)
    for position, index in enumerate(order):
        restored[index] = packed[position]
    return restored


def _stored_ids(vector_store):
    if isinstance(vector_store, SimpleVectorStore):
        return set(vector_store.data.embedding_dict)
    return vector_store.indexed_node_ids()


def _metadata_digest(persist_dir):
    hashes = []
    try:
        for name in ("docstore.json", "index_store.json"):
            with open(os.path.join(persist_dir, name), "rb") as handle:
                hashes.append(hashlib.file_digest(handle, "sha256").hexdigest())
    except FileNotFoundError:
        return None
    return _digest(hashes)


def _preparation_signature(embed_model):
    return _digest({
        "version": 1,
        "embedding": indexing._embed_model_signature(embed_model),
        "max_length": getattr(embed_model, "max_length", None),
        "transformations": [transform.to_dict() for transform in Settings.transformations],
        "metadata_budget": indexing._METADATA_EMBED_BUDGET_CHARS,
        "preparation_code": hashlib.sha256(Path(indexing.__file__).read_bytes()).hexdigest(),
        "normalization_code": hashlib.sha256(Path(indexing.normalize_text.__code__.co_filename).read_bytes()).hexdigest(),
    })


def build_owned_index(documents, vector_store, persist_dir, embed_model, corpus_dir, extensions, rebuild=False):
    if embed_model is None:
        raise ValueError("Owned indexing requires an explicit embedding model.")
    grouped = _source_groups(documents, corpus_dir, extensions)
    store_signature = indexing._vector_store_signature(vector_store)
    store_signature.pop("dim", None)
    owner = _digest([str(Path(corpus_dir).resolve()), store_signature])
    signature = _preparation_signature(embed_model)
    with _ledger(persist_dir, owner) as connection:
        prepared_count = _prepare_cache(connection, grouped, signature, rebuild)
        saved_storage = None
        try:
            saved_storage = StorageContext.from_defaults(
                persist_dir=persist_dir,
                **({} if isinstance(vector_store, SimpleVectorStore) else {"vector_store": vector_store}),
            )
            if isinstance(vector_store, SimpleVectorStore):
                vector_store = saved_storage.vector_store
        except (FileNotFoundError, ValueError, AssertionError, KeyError):
            saved_storage = None
        expected = {row[0] for row in connection.execute("SELECT id FROM children")}
        existing = _stored_ids(vector_store)
        pending = expected if rebuild else expected - existing
        total = len(pending)
        print(f"[index] Owned corpus: {len(expected)} expected child vectors, {total} pending; cached embeddings are reusable.")
        saved_digest = connection.execute("SELECT value FROM config WHERE key='metadata_digest'").fetchone()
        if (not rebuild and not prepared_count and existing == expected and saved_storage is not None
            and saved_digest and saved_digest[0] == _metadata_digest(persist_dir)):
            try:
                saved_structure = saved_storage.index_store.get_index_struct()
                if (saved_structure is not None and set(saved_storage.docstore.docs) == expected
                        and set(saved_structure.nodes_dict.values()) == expected):
                    print("[index] Loaded verified owned index without preparation or embedding.")
                    return VectorStoreIndex(index_struct=saved_structure, storage_context=saved_storage, embed_model=embed_model), saved_storage
            except AssertionError:
                pass
        saved_storage = None
        storage = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=[], storage_context=storage, embed_model=embed_model)
        previous_auto_flush = getattr(vector_store, "auto_flush", None)
        if previous_auto_flush is not None:
            vector_store.auto_flush = False
        try:
            # Page by key rather than holding one cursor open across the whole
            # run: an open read transaction blocks WAL checkpointing, so the WAL
            # grew unboundedly and every read then walked it.
            installed, flushed, last_path, last_ordinal = 0, 0, "", -1
            started = last_report = time.monotonic()
            while True:
                rows = connection.execute(
                    "SELECT id, payload, embedding, dims, path, ordinal FROM children"
                    " WHERE path > ? OR (path = ? AND ordinal > ?)"
                    " ORDER BY path, ordinal LIMIT ?",
                    (last_path, last_path, last_ordinal, _READ_PAGE),
                ).fetchall()
                if not rows:
                    break
                last_path, last_ordinal = rows[-1][4], rows[-1][5]
                page_nodes = [TextNode.from_json(row[1]) for row in rows]
                missing = [(node, row) for node, row in zip(page_nodes, rows) if row[0] in pending]
                need_embedding = []
                for node, row in missing:
                    if row[2] is None:
                        need_embedding.append(node)
                    else:
                        node.embedding = _embedding_values(row[2], row[3])
                if need_embedding:
                    embeddings = _embed_length_bucketed(embed_model, need_embedding)
                    if len(embeddings) != len(need_embedding):
                        raise ValueError("Embedding response is incomplete.")
                    with connection:
                        for node, embedding in zip(need_embedding, embeddings):
                            connection.execute(
                                "UPDATE children SET embedding=?, dims=? WHERE id=?",
                                (_embedding_blob(embedding), len(embedding), node.node_id),
                            )
                            node.embedding = embedding
                if missing:
                    index.insert_nodes([node for node, _ in missing], upsert=True)
                    installed += len(missing)
                    if installed - flushed >= _FLUSH_EVERY:
                        indexing._flush_vector_store(vector_store)
                        flushed = installed
                for node in page_nodes:
                    node.embedding = None  # keep vectors out of the persisted docstore
                    index.index_struct.add_node(node, text_id=node.node_id)
                storage.docstore.add_documents(page_nodes, allow_update=True)
                if total and time.monotonic() - last_report >= _PROGRESS_SECONDS:
                    _report_progress(installed, total, started, persist_dir)
                    last_report = time.monotonic()
            if installed:
                indexing._flush_vector_store(vector_store)
            verified = _stored_ids(vector_store)
            if expected - verified:
                raise RuntimeError("Vector verification is incomplete; checkpoints retained for retry.")
            stale = sorted(verified - expected)
            for start in range(0, len(stale), 256):
                vector_store.delete_nodes(stale[start:start + 256], strict=True)
            if stale:
                indexing._flush_vector_store(vector_store)
                if _stored_ids(vector_store) != expected:
                    raise RuntimeError("Stale-vector deletion is incomplete; retry before querying.")
            storage.index_store.add_index_struct(index.index_struct)
            with Spinner("Persisting verified owned index"):
                storage.persist(persist_dir)
            metadata_digest = _metadata_digest(persist_dir)
            if metadata_digest is None:
                raise RuntimeError("Owned index metadata was not fully persisted; retry before querying.")
            with connection:
                connection.execute("INSERT OR REPLACE INTO config VALUES ('metadata_digest', ?)", (metadata_digest,))
            print(f"[index] Verified {len(expected)} child vectors; removed {len(stale)} obsolete vectors.")
            return index, storage
        finally:
            if previous_auto_flush is not None:
                vector_store.auto_flush = previous_auto_flush