"""Live Milvus validation for Jarvis vector store.

Requires a Milvus server on 127.0.0.1:19530. The test uses an isolated
collection and drops it before and after the run.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from jarvis_helpers import (  # noqa: E402
    build_or_load_persistent_index,
    create_milvus_vector_store,
    _is_port_open,
)
from llama_index.core.embeddings.mock_embed_model import MockEmbedding  # noqa: E402
from llama_index.core.schema import Document  # noqa: E402
from llama_index.core.vector_stores.types import VectorStoreQuery  # noqa: E402


def main() -> int:
    host = os.environ.get("MILVUS_HOST", "127.0.0.1")
    port = os.environ.get("MILVUS_PORT", "19530")
    collection = os.environ.get("MILVUS_TEST_COLLECTION", "jarvis_live_validation")

    print(f"Checking Milvus at {host}:{port}...")
    assert _is_port_open(host, port, timeout=2.0), f"Milvus is not reachable at {host}:{port}"

    embed_model = MockEmbedding(embed_dim=8)
    store = create_milvus_vector_store(collection_name=collection, host=host, port=port, dim=8)

    try:
        store.clear()
    except Exception:
        pass

    docs = [
        Document(
            text="Form W-2 box 1 wages 125000 federal income tax withheld 22000 employer Example Corp",
            metadata={"file_path": "live-w2.pdf", "source": "test", "page_number": 1},
        ),
        Document(
            text="Form 1099 interest income 500 payer Example Bank",
            metadata={"file_path": "live-1099.pdf", "source": "test", "page_number": 2},
        ),
    ]

    try:
        with tempfile.TemporaryDirectory() as persist_dir:
            print("Building persistent index against live Milvus...")
            index, storage_context = build_or_load_persistent_index(
                docs,
                store,
                persist_dir=persist_dir,
                embed_model=embed_model,
                rebuild=True,
            )
            storage_context.persist(persist_dir)

            row_count = store._get_collection_size()
            print(f"Milvus row count after insert: {row_count}")
            assert row_count >= len(docs), "Milvus row count did not reflect inserted documents"

            query_embedding = embed_model._get_query_embedding("W-2 wages withholding")
            result = store.query(VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2))
            print(f"Milvus search ids: {result.ids}")
            assert result.ids, "Milvus search returned no ids"

            reloaded_store = create_milvus_vector_store(collection_name=collection, host=host, port=port, dim=8)
            print("Reloading persistent index metadata against existing Milvus collection...")
            reloaded_index, _ = build_or_load_persistent_index(
                docs,
                reloaded_store,
                persist_dir=persist_dir,
                embed_model=embed_model,
                rebuild=False,
            )
            assert reloaded_index is not None
            assert reloaded_store._get_collection_size() >= len(docs)

        print("Live Milvus validation passed.")
        return 0
    finally:
        try:
            store.clear()
            print(f"Dropped test collection: {collection}")
        except Exception as exc:
            print(f"Warning: failed to drop test collection {collection}: {exc}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"Live Milvus validation failed: {exc}")
        raise SystemExit(2)
    except Exception as exc:
        print(f"Live Milvus validation error: {exc}")
        raise SystemExit(3)
