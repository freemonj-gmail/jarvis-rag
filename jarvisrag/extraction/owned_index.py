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
import sqlite3
import struct
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
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sources (path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS children (
                    id TEXT PRIMARY KEY, path TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    payload TEXT NOT NULL, embedding BLOB
                );
                CREATE INDEX IF NOT EXISTS children_path ON children(path, ordinal);
            """)
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
                    "INSERT INTO children VALUES (?, ?, ?, ?, NULL)",
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
    return struct.pack(f"<{len(vector)}d", *vector)


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
            cursor = connection.execute("SELECT id, payload, embedding FROM children ORDER BY path, ordinal")
            installed = 0
            while rows := cursor.fetchmany(64):
                missing = [row for row in rows if row[0] in pending]
                nodes = [TextNode.from_json(row[1]) for row in missing]
                need_embedding = []
                for node, row in zip(nodes, missing):
                    if row[2] is None:
                        need_embedding.append(node)
                    else:
                        node.embedding = list(struct.unpack(f"<{len(row[2]) // 8}d", row[2]))
                if need_embedding:
                    embeddings = embed_model.get_text_embedding_batch([
                        node.get_content(metadata_mode=MetadataMode.EMBED) for node in need_embedding
                    ])
                    if len(embeddings) != len(need_embedding):
                        raise ValueError("Embedding response is incomplete.")
                    with connection:
                        for node, embedding in zip(need_embedding, embeddings):
                            connection.execute("UPDATE children SET embedding=? WHERE id=?", (_embedding_blob(embedding), node.node_id))
                            node.embedding = embedding
                if nodes:
                    index.insert_nodes(nodes, upsert=True)
                    installed += len(nodes)
                    print(f"[index] Checkpointed {installed}/{total} pending child vectors.")
                for _, payload, _ in rows:
                    node = TextNode.from_json(payload)
                    storage.docstore.add_documents([node], allow_update=True)
                    index.index_struct.add_node(node, text_id=node.node_id)
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