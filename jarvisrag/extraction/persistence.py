"""Persistent vector-index lifecycle for Jarvis.

Index manifest construction, scoped delta-sync and rebuild logic
(``build_or_load_persistent_index``). Chunk preparation lives in
:mod:`jarvisrag.extraction.indexing`; it is kept separate because the mail
ledger hashes that module's source to decide when mail chunks must be
re-prepared, and index-maintenance changes must not force that.
"""
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import run_transformations
from llama_index.core.schema import Document
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from jarvisrag.config import INDEX_MANIFEST_FILE, INDEX_MANIFEST_VERSION, logger
from jarvisrag.extraction.indexing import (
    chunk_documents,
    ensure_document_metadata,
    filter_informative_documents,
    split_large_docs,
)
from jarvisrag.iotools import Spinner

# Number of chunks embedded + inserted into Milvus per delta-sync batch. Batching
# avoids the per-chunk embedding call + Milvus round-trip that one-at-a-time
# index.insert(doc) incurs.
_DELTA_SYNC_BATCH_SIZE = 64


def sync_documents_with_index(index: VectorStoreIndex, documents: List[Document]) -> int:
    try:
        existing_hashes = set(index.docstore.get_all_document_hashes().values())
    except Exception:
        existing_hashes = set()

    new_docs = [doc for doc in documents if doc.hash not in existing_hashes]
    for doc in new_docs:
        index.insert(doc)
    return len(new_docs)


def _count_simple_vector_embeddings(vector_store: BasePydanticVectorStore) -> Optional[int]:
    if not isinstance(vector_store, SimpleVectorStore):
        return None

    data = getattr(vector_store, "data", None)
    embedding_dict = getattr(data, "embedding_dict", None)
    if embedding_dict is None:
        return 0
    return len(embedding_dict)


def _count_storage_context_embeddings(storage_context: StorageContext) -> Optional[int]:
    counts = [
        count
        for count in (
            _count_simple_vector_embeddings(vector_store)
            for vector_store in storage_context.vector_stores.values()
        )
        if count is not None
    ]
    if not counts:
        return None
    return sum(counts)


def _count_storage_context_documents(storage_context: StorageContext) -> Optional[int]:
    docstore = getattr(storage_context, "docstore", None)
    if docstore is None:
        return None
    docs = getattr(docstore, "docs", None)
    if docs is not None:
        try:
            return len(docs)
        except Exception:
            pass
    try:
        hashes = docstore.get_all_document_hashes()
        return len(hashes)
    except Exception:
        return None


def _vector_store_embedding_count(vector_store: BasePydanticVectorStore) -> Optional[int]:
    simple_count = _count_simple_vector_embeddings(vector_store)
    if simple_count is not None:
        return simple_count

    get_collection_size = getattr(vector_store, "_get_collection_size", None)
    if callable(get_collection_size):
        try:
            size = get_collection_size()
            return None if size is None else int(size)
        except Exception:
            return None

    return None


def _index_manifest_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, INDEX_MANIFEST_FILE)


def _indexed_document_ids(vector_store: BasePydanticVectorStore) -> Optional[set[str]]:
    """Ask the vector store which chunks it already holds, or None if it can't say."""
    lookup = getattr(vector_store, "indexed_ref_doc_ids", None)
    if not callable(lookup):
        return None
    try:
        return lookup()
    except Exception as exc:
        logger.warning(f"Could not read indexed documents from the vector store: {exc}")
        return None


def _manifest_without_ids(manifest: Dict[str, Any], excluded_ids: set[str]) -> Dict[str, Any]:
    if not excluded_ids:
        return manifest
    trimmed = dict(manifest)
    trimmed["documents"] = [
        entry for entry in manifest.get("documents", []) if entry.get("id") not in excluded_ids
    ]
    return trimmed


def _embed_model_signature(embed_model: Optional[Any]) -> str:
    if embed_model is None:
        return "default"
    model_name = getattr(embed_model, "model_name", None) or getattr(embed_model, "_model_name", None)
    payload = {
        "class": f"{embed_model.__class__.__module__}.{embed_model.__class__.__name__}",
        "model_name": model_name,
        "query_instruction": getattr(embed_model, "query_instruction", None),
        "text_instruction": getattr(embed_model, "text_instruction", None),
        "normalize": getattr(embed_model, "normalize", None),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _vector_store_signature(vector_store: BasePydanticVectorStore) -> Dict[str, Any]:
    return {
        "class": f"{vector_store.__class__.__module__}.{vector_store.__class__.__name__}",
        "collection_name": getattr(vector_store, "collection_name", None),
        "host": getattr(vector_store, "host", None),
        "port": getattr(vector_store, "port", None),
        "dim": getattr(vector_store, "dim", None),
    }


def _build_index_manifest(
    documents: List[Document],
    vector_store: BasePydanticVectorStore,
    embed_model: Optional[Any],
) -> Dict[str, Any]:
    doc_entries = [
        {
            "id": doc.id_,
            "hash": doc.hash,
            "ref_doc_id": (doc.metadata or {}).get("ref_doc_id"),
            "file_path": (doc.metadata or {}).get("file_path"),
            "chunk_index": (doc.metadata or {}).get("chunk_index"),
        }
        for doc in documents
    ]
    doc_entries.sort(key=lambda item: str(item.get("id")))
    payload = {
        "version": INDEX_MANIFEST_VERSION,
        "embed_model": _embed_model_signature(embed_model),
        "vector_store": _vector_store_signature(vector_store),
        "documents": doc_entries,
    }
    payload["signature"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return payload


def _load_index_manifest(persist_dir: str) -> Optional[Dict[str, Any]]:
    try:
        with open(_index_manifest_path(persist_dir), "r", encoding="utf-8") as file:
            manifest = json.load(file)
        return manifest if isinstance(manifest, dict) else None
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning(f"Unable to read index manifest: {exc}")
        return None


def _write_index_manifest(persist_dir: str, manifest: Dict[str, Any]) -> None:
    try:
        with open(_index_manifest_path(persist_dir), "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True, default=str)
    except Exception as exc:
        logger.warning(f"Unable to write index manifest: {exc}")


def _manifest_matches(current: Dict[str, Any], persisted: Optional[Dict[str, Any]]) -> bool:
    if not persisted:
        return False
    return persisted.get("version") == INDEX_MANIFEST_VERSION and persisted.get("signature") == current.get("signature")


def _scope_file_paths(manifest: Dict[str, Any]) -> set[str]:
    """file_paths present in the current run's manifest. Persisted entries for
    files OUTSIDE this set come from previous runs against other directories
    and must never be touched (no delete, no manifest drop)."""
    return {
        entry.get("file_path")
        for entry in manifest.get("documents", [])
        if entry.get("file_path")
    }


def _is_deleted_file(file_path: Optional[str]) -> bool:
    """An indexed absolute path whose file is gone while its folder remains.

    Requiring the folder keeps an unmounted drive or a moved folder from being
    mistaken for a deletion; relative or synthetic paths are never treated as
    deleted."""
    if not file_path or not os.path.isabs(file_path):
        return False
    return not os.path.exists(file_path) and os.path.isdir(os.path.dirname(file_path))


def _scoped_manifests_match(
    current: Dict[str, Any],
    persisted: Optional[Dict[str, Any]],
    scope_file_paths: set[str],
) -> bool:
    """True when the (id, hash) sets agree for every file in scope_file_paths.
    Used as the fast-path test: if in-scope chunks are unchanged, no Milvus
    work is needed even when out-of-scope persisted entries differ."""
    if not persisted or not scope_file_paths:
        return False
    cur = {
        (entry["id"], entry.get("hash"))
        for entry in current.get("documents", [])
        if entry.get("id") and entry.get("file_path") in scope_file_paths
    }
    per = {
        (entry["id"], entry.get("hash"))
        for entry in persisted.get("documents", [])
        if entry.get("id") and entry.get("file_path") in scope_file_paths
    }
    return cur == per


def _manifest_for_persist(
    current_manifest: Dict[str, Any],
    persisted_manifest: Optional[Dict[str, Any]],
    scope_file_paths: set[str],
) -> Dict[str, Any]:
    """Merge current-run entries with persisted entries from out-of-scope files
    so the written manifest never forgets vectors from previous runs against
    other directories."""
    if not persisted_manifest:
        return current_manifest
    persisted_oos = [
        entry
        for entry in persisted_manifest.get("documents", [])
        if entry.get("file_path") and entry.get("file_path") not in scope_file_paths
    ]
    if not persisted_oos:
        return current_manifest
    merged_entries = list(current_manifest.get("documents", [])) + persisted_oos
    merged_entries.sort(key=lambda item: str(item.get("id")))
    merged = {
        "version": current_manifest.get("version"),
        "embed_model": current_manifest.get("embed_model"),
        "vector_store": current_manifest.get("vector_store"),
        "documents": merged_entries,
    }
    merged["signature"] = hashlib.sha256(
        json.dumps(merged, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return merged


def _structural_signatures_match(
    current: Dict[str, Any], persisted: Optional[Dict[str, Any]]
) -> bool:
    """True when embed model + vector store signatures (and manifest version) match.

    Used to decide whether a delta-sync is safe vs. a full rebuild. The chunk
    set may differ; only the embedding/vector-store schema must be compatible.
    """
    if not persisted:
        return False
    if persisted.get("version") != INDEX_MANIFEST_VERSION:
        return False
    return (
        persisted.get("embed_model") == current.get("embed_model")
        and persisted.get("vector_store") == current.get("vector_store")
    )


def _diff_manifest_documents(
    current: Dict[str, Any],
    persisted: Optional[Dict[str, Any]],
    scope_file_paths: Optional[set[str]] = None,
    indexed_ids: Optional[set[str]] = None,
) -> tuple[list[str], list[str]]:
    """Return (ids_to_insert, ids_to_delete) by comparing chunk id+hash sets.

    When scope_file_paths is provided, persisted chunks whose file_path is NOT
    in that set are ignored — they came from a previous run against a different
    directory and must be preserved.

    When indexed_ids is provided, the vector store itself decides what needs
    inserting. Chunk ids hash the chunk text, so an id already in the store can
    only have been produced by identical content. That makes the store a
    stronger authority than the manifest, which is a file that can go missing,
    move with the code, or record a chunk the store actually rejected.
    """
    current_entries = {
        entry["id"]: entry for entry in current.get("documents", []) if entry.get("id")
    }
    persisted_entries = {
        entry["id"]: entry
        for entry in (persisted or {}).get("documents", [])
        if entry.get("id")
    }
    if scope_file_paths is not None:
        persisted_in_scope = {
            doc_id: entry
            for doc_id, entry in persisted_entries.items()
            if entry.get("file_path") in scope_file_paths
        }
    else:
        persisted_in_scope = persisted_entries
    to_delete = [
        doc_id
        for doc_id, entry in persisted_in_scope.items()
        if doc_id not in current_entries
        or current_entries[doc_id].get("hash") != entry.get("hash")
    ]
    if indexed_ids is not None:
        to_insert = [doc_id for doc_id in current_entries if doc_id not in indexed_ids]
    else:
        to_insert = [
            doc_id
            for doc_id, entry in current_entries.items()
            if doc_id not in persisted_entries
            or persisted_entries[doc_id].get("hash") != entry.get("hash")
        ]
    return to_insert, to_delete


def _stale_stored_ids(
    index: VectorStoreIndex,
    current_manifest: Dict[str, Any],
    scope_file_paths: Optional[set[str]],
) -> list[str]:
    """Stored chunks from in-scope files that the current run no longer produces.

    Asks the docstore and the vector store what they actually hold rather than
    trusting the manifest, which only records the last run and so cannot name
    a stale chunk that an earlier sync failed to remove. Chunks from files
    outside ``scope_file_paths`` belong to other directories and are kept,
    unless their file has since been deleted from disk.
    """
    stored: Dict[str, Optional[str]] = {}
    try:
        for ref_id, info in (index.docstore.get_all_ref_doc_info() or {}).items():
            stored[ref_id] = (getattr(info, "metadata", None) or {}).get("file_path")
    except Exception as exc:
        logger.warning(f"Could not read the docstore's chunk inventory: {exc}")
    lookup = getattr(index.vector_store, "indexed_ref_doc_files", None)
    if callable(lookup):
        try:
            for ref_id, file_path in (lookup() or {}).items():
                stored.setdefault(ref_id, file_path)
        except Exception as exc:
            logger.warning(f"Could not read the vector store's chunk inventory: {exc}")
    current_ids = {entry["id"] for entry in current_manifest.get("documents", []) if entry.get("id")}
    return sorted(
        ref_id
        for ref_id, file_path in stored.items()
        if ref_id not in current_ids
        and (scope_file_paths is None or file_path in scope_file_paths or _is_deleted_file(file_path))
    )


def _apply_manifest_delta(
    index: VectorStoreIndex,
    current_documents: List[Document],
    current_manifest: Dict[str, Any],
    persisted_manifest: Optional[Dict[str, Any]],
    scope_file_paths: Optional[set[str]] = None,
    indexed_ids: Optional[set[str]] = None,
    stale_ids: Optional[List[str]] = None,
) -> tuple[int, int, set[str]]:
    """Insert new/changed chunks and delete removed chunks against an existing
    index. When scope_file_paths is supplied, deletions are restricted to that
    set so vectors indexed by previous runs against other directories survive.

    Emits progress every 25 inserts / 50 deletes so long delta-syncs are never
    silent. Returns the ids that could not be inserted so the caller can keep
    them out of the manifest and retry them on the next run.
    """
    to_insert_ids, to_delete_ids = _diff_manifest_documents(
        current_manifest,
        persisted_manifest,
        scope_file_paths=scope_file_paths,
        indexed_ids=indexed_ids,
    )

    deleted_count = 0
    # A manifest id is the chunk Document's id, which every stored node carries
    # as its ref_doc_id. (metadata["ref_doc_id"] is the FILE path and matches
    # no stored node, so deleting by it silently removed nothing.)
    ref_doc_ids = sorted(set(to_delete_ids) | set(stale_ids or ()))
    if ref_doc_ids:
        total = len(ref_doc_ids)
        print(f"[index] delta-sync: deleting {total} stale chunk(s) from the index...")
        for i, ref_id in enumerate(ref_doc_ids, start=1):
            try:
                index.delete_ref_doc(ref_id, delete_from_docstore=True)
                deleted_count += 1
            except Exception as exc:
                logger.warning(f"Failed to delete stale ref_doc {ref_id}: {exc}")
            if i % 50 == 0 or i == total:
                print(f"[index] delta-sync: deleted {i}/{total} chunk(s)")

    inserted_count = 0
    skipped_count = 0
    skipped_ids: set[str] = set()
    if to_insert_ids:
        insert_set = set(to_insert_ids)
        new_docs = [doc for doc in current_documents if doc.id_ in insert_set]
        total = len(new_docs)
        print(f"[index] delta-sync: embedding {total} new/changed chunk(s)...")
        # Insert in batches so the embedding model and Milvus process many chunks
        # per call. index.insert(doc) embeds a single chunk and does one Milvus
        # round-trip per call, which is dramatically slower on large corpora.
        for start in range(0, total, _DELTA_SYNC_BATCH_SIZE):
            batch = new_docs[start:start + _DELTA_SYNC_BATCH_SIZE]
            try:
                nodes = run_transformations(batch, index._transformations)
                index.insert_nodes(nodes)
                for doc in batch:
                    index.docstore.set_document_hash(doc.id_, doc.hash)
                inserted_count += len(batch)
            except Exception as batch_exc:
                # Fall back to per-doc so a single bad chunk (e.g. metadata too
                # long for Milvus) doesn't drop the whole batch.
                logger.warning(
                    f"Batch insert failed ({batch_exc}); retrying "
                    f"{len(batch)} chunk(s) individually"
                )
                for doc in batch:
                    try:
                        index.insert(doc)
                        inserted_count += 1
                    except Exception as exc:
                        skipped_count += 1
                        skipped_ids.add(doc.id_)
                        meta = doc.metadata or {}
                        logger.warning(
                            f"Skipped chunk {doc.id_} from "
                            f"{meta.get('file_name', meta.get('file_path', '?'))}: {exc}"
                        )
            done = min(start + _DELTA_SYNC_BATCH_SIZE, total)
            print(
                f"[index] delta-sync: embedded {done}/{total} chunk(s) "
                f"(skipped={skipped_count})"
            )

    return inserted_count, deleted_count, skipped_ids


def _is_owned_storage_file(name: str) -> bool:
    """Files a LlamaIndex StorageContext (plus our manifest) writes into persist_dir."""
    return name in {"docstore.json", "index_store.json", "graph_store.json", INDEX_MANIFEST_FILE} or (
        name.endswith("vector_store.json")
    )


def _reset_persistent_storage(persist_dir: str, vector_store: BasePydanticVectorStore) -> StorageContext:
    # Preserve the per-file extraction cache, other indexes' storage (domain
    # corpora persist under domains/) and anything else beside the index.
    # Only delete files known to belong to the LlamaIndex storage context +
    # our manifest, NOT the whole directory tree.
    if os.path.isdir(persist_dir):
        for name in os.listdir(persist_dir):
            full = os.path.join(persist_dir, name)
            if not _is_owned_storage_file(name) or not os.path.isfile(full):
                continue
            try:
                os.remove(full)
            except OSError as exc:
                logger.warning(f"Could not remove {full} during persistent storage reset: {exc}")
    os.makedirs(persist_dir, exist_ok=True)
    return StorageContext.from_defaults(vector_store=vector_store)


def _clear_vector_store(vector_store: BasePydanticVectorStore) -> None:
    if hasattr(vector_store, "clear"):
        try:
            vector_store.clear()
        except Exception as e:
            logger.warning(f"Unable to clear vector store before rebuild: {e}")


def _flush_vector_store(vector_store: BasePydanticVectorStore) -> None:
    flush = getattr(vector_store, "flush", None)
    if callable(flush):
        flush()


def build_or_load_persistent_index(
    documents: List[Document],
    vector_store: BasePydanticVectorStore,
    persist_dir: str,
    embed_model: Optional[Any] = None,
    sync_existing: bool = False,
    rebuild: bool = False,
    *,
    owned_corpus_dir: Optional[str] = None,
    owned_extensions: tuple[str, ...] = (),
) -> tuple[VectorStoreIndex, StorageContext]:
    if owned_corpus_dir is not None:
        from jarvisrag.extraction.owned_index import build_owned_index

        return build_owned_index(
            documents, vector_store, persist_dir, embed_model,
            owned_corpus_dir, owned_extensions, rebuild=rebuild,
        )
    # These four steps are silent CPU loops over every document; on a large
    # corpus they look like a hang, so surface live status + per-phase timing.
    with Spinner(f"Preparing {len(documents)} documents") as spin:
        for doc in documents:
            ensure_document_metadata(doc)
        spin.update("Splitting into chunks")
        split_documents = split_large_docs(documents)
        spin.update(f"Filtering low-information chunks (from {len(split_documents)})")
        index_documents = filter_informative_documents(split_documents)
        spin.update(f"Building manifest signature ({len(index_documents)} chunks)")
        current_manifest = _build_index_manifest(index_documents, vector_store, embed_model)

    if not os.path.isdir(persist_dir):
        os.makedirs(persist_dir, exist_ok=True)

    if rebuild:
        print("[index] Rebuild requested: clearing vector store and persistent storage...")
        _clear_vector_store(vector_store)
        storage_context = _reset_persistent_storage(persist_dir, vector_store)
    else:
        storage_context = None

    if storage_context is None:
        # Reloading a large docstore from disk is silent; show live status.
        with Spinner("Loading persistent storage"):
            is_simple_vector_store = isinstance(vector_store, SimpleVectorStore)
            try:
                if is_simple_vector_store:
                    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                else:
                    storage_context = StorageContext.from_defaults(
                        vector_store=vector_store,
                        persist_dir=persist_dir,
                    )
            except (FileNotFoundError, AssertionError, ValueError, OSError) as exc:
                print(f"[index] No usable persistent storage on disk ({type(exc).__name__}); starting fresh storage context.")
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                )

    print("[index] Comparing current document signature against persisted manifest...")
    persisted_manifest = _load_index_manifest(persist_dir)
    scope_file_paths = _scope_file_paths(current_manifest)
    # A file deleted from disk is in no run's documents, so it would never come
    # into scope; add it so its chunks and manifest entries are removed.
    scope_file_paths |= {
        entry.get("file_path")
        for entry in (persisted_manifest or {}).get("documents", [])
        if _is_deleted_file(entry.get("file_path"))
    }
    structural_ok = _structural_signatures_match(current_manifest, persisted_manifest)
    # The scoped fast path must NEVER skip a rebuild when the embed model or
    # vector store schema changed — old vectors would be queried with a new
    # embedding model and return garbage. Gate it on structural_ok.
    manifest_ok = _manifest_matches(current_manifest, persisted_manifest) or (
        structural_ok
        and _scoped_manifests_match(current_manifest, persisted_manifest, scope_file_paths)
    )
    if not rebuild and not manifest_ok and not structural_ok:
        if persisted_manifest:
            print(
                "Persistent index manifest is structurally incompatible "
                "(embed model or vector store signature changed). "
                "Rebuilding active vector store and local metadata..."
            )
        else:
            print(
                "Persistent index manifest is missing. "
                "Building active vector store and local metadata..."
            )
        print("[index] Clearing existing vector store contents (this can take a while for large Milvus collections)...")
        _clear_vector_store(vector_store)
        print("[index] Resetting the index metadata files (caches and other indexes are kept)...")
        storage_context = _reset_persistent_storage(persist_dir, vector_store)

    index_struct = None
    try:
        index_struct = storage_context.index_store.get_index_struct()
    except AssertionError:
        index_struct = None

    is_collection_store = (
        hasattr(vector_store, "collection_name")
        and hasattr(vector_store, "has_collection")
    )
    collection_exists = True
    if is_collection_store:
        try:
            collection_exists = bool(vector_store.has_collection())
        except Exception:
            collection_exists = False

    if index_struct is not None and collection_exists:
        persisted_embedding_count = _vector_store_embedding_count(vector_store)
        if persisted_embedding_count is None:
            persisted_embedding_count = _count_storage_context_embeddings(storage_context)
        expected_embedding_count = _count_storage_context_documents(storage_context)
        if expected_embedding_count is None:
            expected_embedding_count = len(index_documents)
        count_mismatch = (
            persisted_embedding_count is not None
            and persisted_embedding_count != expected_embedding_count
        )
        # A store that can list what it holds is reconciled chunk by chunk;
        # wiping it would also drop vectors other directories indexed into it.
        can_reconcile = callable(getattr(vector_store, "indexed_ref_doc_files", None))
        if count_mismatch and not can_reconcile:
            store_label = "Milvus collection" if is_collection_store else "local vector store"
            print(
                f"Persistent index metadata found, but {store_label} has "
                f"{persisted_embedding_count}/{expected_embedding_count} vectors. "
                "Rebuilding vector store from source documents..."
            )
            _clear_vector_store(vector_store)
            storage_context = _reset_persistent_storage(persist_dir, vector_store)
            index_struct = None
        else:
            if count_mismatch:
                print(
                    f"[index] vector store holds {persisted_embedding_count} vectors but the "
                    f"local metadata lists {expected_embedding_count}; reconciling chunk by chunk."
                )
                manifest_ok = False
            index = VectorStoreIndex(
                index_struct=index_struct,
                storage_context=storage_context,
                embed_model=embed_model,
            )
            # A matching manifest only proves the chunk set is unchanged, not
            # that every chunk reached the store, so confirm against the store
            # before taking the fast path.
            indexed_ids = _indexed_document_ids(vector_store)
            if manifest_ok and indexed_ids is not None:
                absent = [
                    entry["id"]
                    for entry in current_manifest.get("documents", [])
                    if entry.get("id") and entry["id"] not in indexed_ids
                ]
                if absent:
                    print(
                        f"[index] manifest matches but {len(absent)} chunk(s) are "
                        "missing from the vector store; syncing them."
                    )
                    manifest_ok = False
            # The reverse check: chunks the store still holds for this corpus
            # that the current documents no longer produce.
            stale_ids = _stale_stored_ids(index, current_manifest, scope_file_paths)
            if manifest_ok and stale_ids:
                print(
                    f"[index] manifest matches but {len(stale_ids)} stale chunk(s) "
                    "remain in the index; removing them."
                )
                manifest_ok = False
            if manifest_ok:
                print("Loaded existing persistent index.")
            else:
                inserted, deleted, skipped_ids = _apply_manifest_delta(
                    index,
                    index_documents,
                    current_manifest,
                    persisted_manifest,
                    scope_file_paths=scope_file_paths,
                    indexed_ids=indexed_ids,
                    stale_ids=stale_ids,
                )
                # A skipped chunk never reached the store, so recording it would
                # make the next run treat it as done and lose it permanently.
                persistable_manifest = _manifest_without_ids(current_manifest, skipped_ids)
                if inserted or deleted:
                    print(
                        f"Delta-synced persistent index: {inserted} chunk(s) added, "
                        f"{deleted} removed (scoped to current run; other-directory "
                        "vectors preserved)."
                    )
                    _flush_vector_store(vector_store)
                    storage_context.persist(persist_dir)
                    _write_index_manifest(
                        persist_dir,
                        _manifest_for_persist(persistable_manifest, persisted_manifest, scope_file_paths),
                    )
                else:
                    print("Loaded existing persistent index.")
                    _write_index_manifest(
                        persist_dir,
                        _manifest_for_persist(persistable_manifest, persisted_manifest, scope_file_paths),
                    )
            if sync_existing:
                print("Syncing existing index with current documents...")
                inserted = sync_documents_with_index(index, index_documents)
                if inserted:
                    print(f"Inserted {inserted} new documents into existing index.")
                    storage_context.persist(persist_dir)
                    _write_index_manifest(
                        persist_dir,
                        _manifest_for_persist(current_manifest, persisted_manifest, scope_file_paths),
                    )
            return index, storage_context

    if index_struct is not None and not collection_exists:
        print("Persistent index metadata found, but Milvus collection is missing. Rebuilding Milvus collection from source documents...")

    index = None
    docs_per_batch = int(os.environ.get("JARVIS_INDEX_BATCH_SIZE", "64"))
    docs_per_batch = max(docs_per_batch, 1)
    batches = chunk_documents(index_documents, docs_per_chunk=docs_per_batch)
    total_batches = len(batches)
    print(
        f"Embedding {len(index_documents)} chunks into the vector store in "
        f"{total_batches} batch(es) of up to {docs_per_batch}. This is the slow step "
        "on first build; subsequent runs reuse the index when the manifest matches."
    )
    previous_auto_flush = getattr(vector_store, "auto_flush", None)
    if previous_auto_flush is not None:
        setattr(vector_store, "auto_flush", False)
    skipped_in_rebuild = 0
    try:
        for batch_number, batch in enumerate(batches, start=1):
            if not batch:
                continue
            print(f"Indexing batch {batch_number}/{total_batches} ({len(batch)} documents)...")
            if index is None:
                # nodes=[] satisfies VectorStoreIndex's "one of nodes/objects/index_struct"
                # requirement without a from_documents call that has no per-batch guard.
                index = VectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context,
                    embed_model=embed_model,
                )
            try:
                # Batched embed+insert (same path as delta-sync): the embedding model
                # processes the whole batch per forward pass instead of one chunk at a
                # time — dramatically faster on large first builds.
                nodes = run_transformations(batch, index._transformations)
                index.insert_nodes(nodes)
                for doc in batch:
                    index.docstore.set_document_hash(doc.id_, doc.hash)
            except Exception as batch_exc:
                # Fall back to per-doc so a single bad chunk can't drop the batch.
                logger.warning(
                    f"Batch insert failed ({batch_exc}); retrying "
                    f"{len(batch)} chunk(s) individually"
                )
                for doc in batch:
                    try:
                        index.insert(doc)
                    except Exception as exc:
                        skipped_in_rebuild += 1
                        meta = doc.metadata or {}
                        logger.warning(
                            f"Skipped chunk {doc.id_} from "
                            f"{meta.get('file_name', meta.get('file_path', '?'))}: {exc}"
                        )

        if index is None:
            index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                embed_model=embed_model,
            )
        if skipped_in_rebuild:
            print(
                f"[index] Rebuild completed with {skipped_in_rebuild} chunk(s) skipped "
                "(see warnings above)."
            )
        _flush_vector_store(vector_store)
        storage_context.persist(persist_dir)
        _write_index_manifest(
            persist_dir,
            _manifest_for_persist(current_manifest, persisted_manifest, scope_file_paths),
        )
    finally:
        if previous_auto_flush is not None:
            setattr(vector_store, "auto_flush", previous_auto_flush)

    return index, storage_context
