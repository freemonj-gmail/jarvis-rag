"""Milvus-backed vector store for Jarvis.

Provides :class:`MilvusVectorStore` (a LlamaIndex ``BasePydanticVectorStore``
implementation targeting the Milvus Community Edition HTTP client), the
embedding-normalisation helper, metadata-filter translation, and the
``create_milvus_vector_store`` factory.

Depends on the standard library, numpy, pymilvus, LlamaIndex vector-store
types, and :mod:`jarvis_config` for the shared logger.
"""
import json
import time
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Set

import numpy as np
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from jarvisrag.config import logger


def normalize_vector(embedding: Sequence[float]) -> List[float]:
    vector = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        return (vector / norm).tolist()
    return vector.tolist()


def _metadata_filters_to_milvus_expr(filters: Optional[MetadataFilters]) -> Optional[str]:
    if filters is None:
        return None

    try:
        filter_list = filters.legacy_filters()
    except Exception as exc:
        raise ValueError(
            "Milvus vector store only supports exact-match metadata filters."
        ) from exc

    expr_pieces: List[str] = []
    for filt in filter_list:
        key = filt.key
        value = filt.value
        if key not in {"ref_doc_id", "source", "file_path", "parent_file"}:
            raise ValueError(
                "Milvus vector store only supports filters on ref_doc_id, source, file_path, or parent_file."
            )

        if isinstance(value, str):
            safe_value = value.replace('"', '\\"')
            expr_pieces.append(f'{key} == "{safe_value}"')
        else:
            expr_pieces.append(f"{key} == {value}")

    condition = filters.condition.value if getattr(filters.condition, 'value', None) else 'and'
    return f" {condition} ".join(expr_pieces)


# Exact search over every stored vector. Measured on the mail corpus (~330k
# bge-m3 vectors): HNSW (M=16, ef=64) returned only ~68% of the true top-32 for
# natural-language questions, while FLAT is exact at ~190 ms per search.
_EXACT_INDEX_TYPE = "FLAT"


def _exact_index_params() -> Any:
    from pymilvus.milvus_client.index import IndexParams

    idx = IndexParams()
    idx.add_index(
        field_name="embedding",
        index_type=_EXACT_INDEX_TYPE,
        index_name="embedding_flat",
        metric_type="IP",
    )
    return idx


class MilvusVectorStore(BasePydanticVectorStore):
    stores_text: bool = False
    is_embedding_query: bool = True
    auto_flush: bool = True

    collection_name: str = "jarvis_context"
    host: str = "127.0.0.1"
    port: str = "19530"
    dim: Optional[int] = None
    _collection: Optional[Any] = PrivateAttr(default=None)
    _client: Optional[MilvusClient] = PrivateAttr(default=None)

    def __init__(
        self,
        collection_name: str = "jarvis_context",
        host: str = "127.0.0.1",
        port: str = "19530",
        dim: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(collection_name=collection_name, host=host, port=port, dim=dim, **kwargs)
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.dim = dim
        self._collection = None
        self._client = None

    @property
    def client(self) -> Optional[Any]:
        return self._collection

    def has_collection(self) -> bool:
        self._connect()
        return self._client is not None and self._client.has_collection(self.collection_name)

    def _connect(self) -> None:
        if self._client is not None:
            return
        try:
            # Prefer the new MilvusClient HTTP interface
            self._client = MilvusClient(f"http://{self.host}:{self.port}")

        except Exception as exc:
            # Try fallback port if user mistakenly provided 2379
            if self.port == "2379":
                try:
                    self._client = MilvusClient(f"http://{self.host}:19530")
                    self.port = "19530"
                    return
                except Exception:
                    pass
            raise RuntimeError(f"Failed to connect to Milvus at {self.host}:{self.port}: {exc}") from exc

    def _ensure_collection(self, dim: int) -> None:
        self._connect()

        if self._client is None:
            raise RuntimeError("Milvus client is not initialized.")

        # -------------------------------------------------
        # EXISTING COLLECTION
        # -------------------------------------------------
        if self._client.has_collection(self.collection_name):

            try:
                desc = self._client.describe_collection(self.collection_name)

                fields = desc.get("fields", []) if isinstance(desc, dict) else []

                vector_field = next(
                    (f for f in fields if f.get("name") == "embedding"),
                    None,
                )

                if vector_field is None:
                    raise RuntimeError(
                        "Milvus collection schema does not contain an embedding field."
                    )

                self.dim = int(vector_field.get("params", {}).get("dim"))
                if dim and self.dim != dim:
                    raise RuntimeError(
                        f"Milvus collection embedding dimension is {self.dim}, "
                        f"but the active embedding model produced {dim}. "
                        "Use a matching embedding model or rebuild/drop the collection."
                    )

            except Exception as e:
                if self.dim is None:
                    raise RuntimeError(
                        f"Could not determine embedding dimension: {e}"
                    ) from e
                raise

            # ---------------------------------------------
            # ENSURE THE EXACT (FLAT) INDEX; REPLACE AN OLDER ANN INDEX IN PLACE
            # ---------------------------------------------
            try:
                existing_indexes = self._client.list_indexes(self.collection_name)
                existing_types = {
                    name: str(self._client.describe_index(self.collection_name, name).get("index_type", "")).upper()
                    for name in existing_indexes
                }

                if any(index_type != _EXACT_INDEX_TYPE for index_type in existing_types.values()):
                    print(
                        f"[jarvis] Switching Milvus collection '{self.collection_name}' to exact "
                        f"vector search (was {', '.join(sorted(set(existing_types.values())))}); "
                        "stored vectors are kept."
                    )
                    self._client.release_collection(self.collection_name)
                    for name, index_type in existing_types.items():
                        if index_type != _EXACT_INDEX_TYPE:
                            self._client.drop_index(self.collection_name, name)
                            existing_indexes = [n for n in existing_indexes if n != name]

                if not existing_indexes:
                    self._client.create_index(
                        collection_name=self.collection_name,
                        index_params=_exact_index_params(),
                    )
                    logger.info(
                        f"Created {_EXACT_INDEX_TYPE} index on collection {self.collection_name}"
                    )

            except Exception as e:
                logger.warning(f"Index verification failed: {e}")

            try:
                self._client.load_collection(self.collection_name)
            except Exception as e:
                logger.warning(f"Failed to load collection: {e}")

            self._collection = self._client
            return

        # -------------------------------------------------
        # CREATE NEW COLLECTION
        # -------------------------------------------------
        self.dim = dim

        if self.dim is None:
            raise ValueError(
                "Milvus collection does not exist and no embedding dimension was provided."
            )

        fields = [
            FieldSchema(
                name="node_id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=1024,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            ),
            FieldSchema(
                name="ref_doc_id",
                dtype=DataType.VARCHAR,
                max_length=1024,
            ),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=1024,
            ),
            FieldSchema(
                name="file_path",
                dtype=DataType.VARCHAR,
                max_length=2048,
            ),
            FieldSchema(
                name="parent_file",
                dtype=DataType.VARCHAR,
                max_length=2048,
            ),
            FieldSchema(
                name="metadata_json",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
        ]

        schema = CollectionSchema(
            fields,
            description="Jarvis Milvus vector store",
        )

        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )

        # -------------------------------------------------
        # CREATE INDEX
        # -------------------------------------------------
        self._client.create_index(
            collection_name=self.collection_name,
            index_params=_exact_index_params(),
        )

        self._client.load_collection(self.collection_name)

        logger.info(
            f"Created Milvus collection and index: {self.collection_name}"
        )

        self._collection = self._client

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        raise NotImplementedError("MilvusVectorStore does not store text nodes directly.")

    def _get_collection_size(self) -> int:
        """Get the current number of vectors in the collection."""
        try:
            self._connect()
        except Exception:
            return 0

        # Prefer client stats when available
        try:
            if self._client is not None and self._client.has_collection(self.collection_name):
                stats = self._client.get_collection_stats(self.collection_name)
                return int(stats.get("row_count", 0))
        except Exception:
            pass
        # Fallback to 0 if unknown
        return 0

    def indexed_ref_doc_ids(self) -> Optional[Set[str]]:
        """Every ``ref_doc_id`` currently stored, or None if it can't be read.

        Callers use this to ask the collection itself what is already indexed
        instead of trusting bookkeeping kept beside the code. Returning None
        (rather than an empty set) keeps "unknown" distinct from "nothing
        indexed", so a transient failure can never be read as an empty index.
        """
        try:
            self._connect()
        except Exception as exc:
            logger.warning(f"Could not connect to Milvus to list indexed documents: {exc}")
            return None
        if self._client is None or not self._client.has_collection(self.collection_name):
            return None

        ref_doc_ids: Set[str] = set()
        try:
            iterator = self._client.query_iterator(
                collection_name=self.collection_name,
                filter='node_id != ""',
                output_fields=["ref_doc_id"],
                batch_size=1000,
            )
            while True:
                batch = iterator.next()
                if not batch:
                    break
                ref_doc_ids.update(
                    row["ref_doc_id"] for row in batch if row.get("ref_doc_id")
                )
            iterator.close()
        except Exception as exc:
            logger.warning(f"Could not enumerate indexed documents from Milvus: {exc}")
            return None
        return ref_doc_ids

    def indexed_node_ids(self) -> Set[str]:
        """Enumerate actual children; a failed read must not look like an empty store."""
        self._connect()
        if self._client is None:
            raise RuntimeError("Milvus client is unavailable.")
        if not self._client.has_collection(self.collection_name):
            return set()
        iterator = self._client.query_iterator(
            collection_name=self.collection_name, filter='node_id != ""',
            output_fields=["node_id"], batch_size=1000, consistency_level="Strong",
        )
        identities: Set[str] = set()
        try:
            while rows := iterator.next():
                identities.update(str(row["node_id"]) for row in rows)
        finally:
            iterator.close()
        return identities

    def add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:

        if not nodes:
            return []

        if self._collection is None:
            self._ensure_collection(dim=len(nodes[0].get_embedding()))

        rows = []

        for node in nodes:

            metadata = dict(node.metadata or {})
            embedding = normalize_vector(node.get_embedding())
            if self.dim is not None and len(embedding) != self.dim:
                raise RuntimeError(
                    f"Node {node.node_id} has embedding dimension {len(embedding)}, "
                    f"but Milvus collection expects {self.dim}."
                )

            if "insertion_timestamp" not in metadata:
                metadata["insertion_timestamp"] = (
                    datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"
                )

            rows.append({
                "node_id": node.node_id,
                "embedding": embedding,
                "ref_doc_id": str(
                    node.ref_doc_id
                    or metadata.get("ref_doc_id")
                    or metadata.get("file_path")
                    or node.node_id
                ),
                "source": str(metadata.get("source", "file")),
                "file_path": str(metadata.get("file_path", "")),
                "parent_file": str(metadata.get("parent_file", "")),
                "metadata_json": json.dumps(metadata, default=str),
            })

        try:

            if self._client is not None and self._client.has_collection(
                self.collection_name
            ):

                write = self._client.upsert if kwargs.get("upsert", False) else self._client.insert
                write(
                    self.collection_name,
                    rows,
                )
                if self.auto_flush:
                    self.flush()

            else:
                raise RuntimeError(
                    "Milvus collection is unavailable."
                )

        except Exception as e:
            raise RuntimeError(f"Milvus insert failed: {e}") from e

        return [row["node_id"] for row in rows]

    def flush(self) -> None:
        self._connect()
        if self._client is None or not self._client.has_collection(self.collection_name):
            return
        try:
            self._client.flush(collection_name=self.collection_name)
        except TypeError:
            self._client.flush(self.collection_name)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._connect()
        expr = f'ref_doc_id == "{ref_doc_id.replace("\"", "\\\"")}"'
        if self._client is not None and self._client.has_collection(self.collection_name):
            try:
                self._client.delete(self.collection_name, filter=expr)
            except Exception as e:
                logger.warning(f"Milvus client delete failed: {e}")
        else:
            raise RuntimeError("Milvus client is not initialized or collection does not exist; cannot perform delete.")

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        self._connect()

        expr: Optional[str] = None
        if node_ids is not None:
            quoted_ids = ", ".join([f'"{nid.replace("\"", "\\\"")}"' for nid in node_ids])
            expr = f"node_id in [{quoted_ids}]"
        elif filters is not None:
            expr = _metadata_filters_to_milvus_expr(filters)

        if expr is None and node_ids is None:
            raise ValueError("MilvusVectorStore.delete_nodes requires node_ids or metadata filters.")

        if self._client is not None and self._client.has_collection(self.collection_name):
            try:
                if node_ids is not None:
                    self._client.delete(self.collection_name, ids=node_ids)
                else:
                    self._client.delete(self.collection_name, filter=expr)
            except Exception as e:
                logger.warning(f"Milvus client delete_nodes failed: {e}")
                if delete_kwargs.get("strict", False):
                    raise
        else:
            raise RuntimeError("Milvus client is not initialized or collection does not exist; cannot perform delete_nodes.")

    def clear(self) -> None:
        self._connect()
        if self._client is not None and self._client.has_collection(self.collection_name):
            try:
                self._client.drop_collection(self.collection_name)
            except Exception as e:
                logger.warning(f"Milvus client drop_collection failed: {e}")
        else:
            if self._collection is not None:
                try:
                    self._collection.drop()
                except Exception:
                    pass
        self._collection = None

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        if query.query_embedding is None:
            raise ValueError("MilvusVectorStore requires a query embedding.")

        self._connect()
        if self._client is None or not self._client.has_collection(self.collection_name):
            raise RuntimeError("Milvus collection does not exist or client is not initialized.")
        if self._collection is None:
            # A warm start never calls add(), so the exact-index check runs on first search.
            self._ensure_collection(dim=self.dim or 0)

        query_embedding = normalize_vector(query.query_embedding)
        if self.dim is not None and len(query_embedding) != self.dim:
            raise RuntimeError(
                f"Query embedding dimension is {len(query_embedding)}, "
                f"but Milvus collection expects {self.dim}."
            )
        expr = _metadata_filters_to_milvus_expr(query.filters)

        search_params = {"metric_type": "IP"}
        # Prefer client.search and retry once for transient QueryNode/channel issues.
        top_ids: List[str] = []
        top_similarities: List[float] = []
        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                if self._client is None or not self._client.has_collection(self.collection_name):
                    raise RuntimeError("Milvus collection is unavailable for search.")

                results = self._client.search(
                    self.collection_name,
                    data=[query_embedding],
                    anns_field="embedding",
                    search_params=search_params,
                    limit=query.similarity_top_k,
                    filter=expr or "",
                    output_fields=["node_id"],
                )

                if results and len(results) > 0 and results[0]:
                    for hit in results[0]:
                        nid = hit.get("node_id") or (hit.get("entity") or {}).get("node_id")
                        dist = hit.get("distance")
                        if nid is not None and dist is not None:
                            top_ids.append(str(nid))
                            top_similarities.append(float(dist))

                return VectorStoreQueryResult(ids=top_ids, similarities=top_similarities)
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                transient = (
                    "channel distribution is not serviceable" in msg
                    or "channel not available" in msg
                    or "code=503" in msg
                )
                if attempt == 0 and transient:
                    logger.warning(
                        f"Transient Milvus search failure. Retrying after reload: {e}"
                    )
                    try:
                        if self._client is not None and self._client.has_collection(self.collection_name):
                            self._client.load_collection(self.collection_name)
                    except Exception as load_exc:
                        logger.warning(f"Milvus reload before retry failed: {load_exc}")
                    time.sleep(0.5)
                    continue
                break

        raise RuntimeError(f"Milvus search failed: {last_error}")

    def persist(
        self,
        persist_path: str,
        fs: Optional[Any] = None,
    ) -> None:
        return None


def create_milvus_vector_store(
    collection_name: str = "jarvis_context",
    host: str = "127.0.0.1",
    port: str = "19530",
    dim: Optional[int] = None,
) -> MilvusVectorStore:
    return MilvusVectorStore(collection_name=collection_name, host=host, port=port, dim=dim)
