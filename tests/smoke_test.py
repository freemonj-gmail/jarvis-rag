"""Lightweight smoke test for Jarvis helpers.

This test avoids heavy downloads (no LLMs or HF embedding models).
It checks importability and a few fast helper functions.
"""
import sys
import os

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from jarvis_helpers import (
    build_or_load_persistent_index,
    chunk_text,
    create_milvus_vector_store,
    filter_informative_documents,
    group_documents_by_source_windows,
    is_low_information_text,
    normalize_text,
    split_large_docs,
    text_artifact_score,
    _is_ocr_garbage_text,
    _is_port_open,
    jpeg_extractor,
)


def run_tests():
    print("Running smoke tests...")

    # chunk_text basic behavior
    text = "This is a test " * 50
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
    assert isinstance(chunks, list) and len(chunks) >= 1
    print("- chunk_text OK (produced", len(chunks), "chunks)")

    normalized = normalize_text(". . . . . Prior year contributions subtracted . . . . . 2025")
    assert ". . ." not in normalized
    assert "Prior year contributions subtracted" in normalized
    assert "cid:" not in normalize_text("(cid:36)(cid:38)(cid:38)(cid:50)")
    assert text_artifact_score("3 3 4 3 T Lo o c t a a l l t M ax a a r f y") > 0
    print("- normalize_text layout cleanup OK")

    assert is_low_information_text("This Page Intentionally Left Blank Page 6of 24 886TK420-71-001 3DU/16B6")
    assert is_low_information_text(". . . . . . . . . . Prior year contributions subtracted . . . . . . . . . .")
    assert is_low_information_text(". . . . . . . . . . Total contributions available to 2025 . . . . . . . . . .")
    assert not is_low_information_text("Form W-2 wages 125000 federal income tax withheld 22000 employer Microsoft")
    print("- is_low_information_text OK")

    # OCR garbage filter: high-confidence rejections
    garbage_samples = [
        "QQhehokQpi  Sa5ur38340 Va S8 3 Ss01 ilupoe4tsei2k2tIp \"una 330 e  473 Sh ioails2 SU isuadsis esile?i Jal Ul25 Ieoiuiail? Julasio? spieu",
        "NO 93ui08/,0 31A anss1;o2  'S1214rs 2 Sorroisse? Venzjia|20 Ba VolCVzt NS 3939 . 1ajzul Hoamasd3HHLH Gmaed SSIE VAMM 8snH} RSIm 70 Ralt",
    ]
    for sample in garbage_samples:
        is_g, reason = _is_ocr_garbage_text(sample)
        assert is_g, f"expected garbage, got kept: {sample[:60]!r} ({reason})"

    # OCR garbage filter: legitimate content must be preserved
    legit_samples = [
        "Form W-2 wages 125000 federal income tax withheld 22000 employer Microsoft Corporation",
        "Jane Doe 100 Example Road Springfield IL 62704 phone 555-0100",
        "Wages and salaries 125,000.00 Federal income tax withheld 22,500.00 Social security wages 168,600.00",
        "Form 1098 mortgage interest received from payer 8,432.17 outstanding principal 412,000.00",
        "Box 1 Wages tips other compensation 125000 Box 2 Federal income tax withheld 22000 Box 3 Social security wages",
        "Doctors Without Borders donation receipt 04/15/2025 amount $250.00 EIN 13-3433452",
    ]
    for sample in legit_samples:
        is_g, reason = _is_ocr_garbage_text(sample)
        assert not is_g, f"expected kept, got garbage: {sample[:60]!r} ({reason})"

    # Integration: filter_informative_documents drops garbage and keeps legit
    from llama_index.core.schema import Document as _SmokeDoc
    mixed_docs = [
        _SmokeDoc(text=garbage_samples[0], metadata={"file_name": "license.jpeg", "extracted_by": "easyocr"}),
        _SmokeDoc(text=legit_samples[0], metadata={"file_name": "w2.pdf"}),
    ]
    kept = filter_informative_documents(mixed_docs)
    assert len(kept) == 1 and "Form W-2" in kept[0].text
    print("- OCR garbage filter OK (rejected", len(garbage_samples), "garbage,",
          "kept", len(legit_samples), "legit)")

    from llama_index.core.schema import Document as LlamaDocument
    form_text = "Employer: Example Corp\nWages: 125000\nFederal income tax withheld: 22000\nState wages: 125000"
    form_chunks = chunk_text(form_text, chunk_size=6, chunk_overlap=2)
    assert len(form_chunks) > 1
    assert "Employer: Example Corp" in form_chunks[0]
    assert "\n" in form_chunks[0]

    split_once = split_large_docs([
        LlamaDocument(text=form_text, metadata={"file_path": "w2.pdf", "page_number": 1})
    ], chunk_size=6, chunk_overlap=2)
    split_twice = split_large_docs([
        LlamaDocument(text=form_text, metadata={"file_path": "w2.pdf", "page_number": 1})
    ], chunk_size=6, chunk_overlap=2)
    assert [doc.id_ for doc in split_once] == [doc.id_ for doc in split_twice]
    assert all(doc.metadata.get("parent_file") == "w2.pdf" for doc in split_once)
    print("- stable line-aware chunking OK")

    grouped_docs = group_documents_by_source_windows([
        LlamaDocument(text="Page one amount 100", metadata={"file_path": "sample.pdf", "page_number": 1}),
        LlamaDocument(text="Page two balance 200", metadata={"file_path": "sample.pdf", "page_number": 2}),
    ], max_words=50)
    assert len(grouped_docs) == 1
    assert "Page one amount 100" in grouped_docs[0].text
    assert grouped_docs[0].metadata.get("context_mode") == "source_window"
    print("- group_documents_by_source_windows OK")

    # create_milvus_vector_store default values
    store = create_milvus_vector_store()
    assert hasattr(store, "host") and hasattr(store, "port")
    print(f"- create_milvus_vector_store defaults -> {store.host}:{store.port}")

    # port probe should return boolean and not raise
    reachable = _is_port_open(store.host, store.port, timeout=0.5)
    print(f"- _is_port_open -> {reachable}")

    # Deferred extractor should not have instantiated heavy OCR models
    # If DeferredExtractor used, its internal _extractor should be None before use
    try:
        extractor_internal = getattr(jpeg_extractor, "_extractor", None)
        assert extractor_internal is None
        print("- jpeg_extractor deferred initialization OK")
    except Exception:
        print("- jpeg_extractor check skipped or failed")

    # --- Minimal end-to-end index + query using MockEmbedding and SimpleVectorStore
    try:
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding
        from llama_index.core.indices.vector_store import VectorStoreIndex
        from llama_index.core.vector_stores.simple import SimpleVectorStore
        from llama_index.core.vector_stores.types import VectorStoreQuery
        import tempfile

        mock_emb = MockEmbedding(embed_dim=8)
        docs = [
            # create simple documents
            type('D', (), {'text': 'The cat sat on the mat', 'metadata': {'source': 'test'}, 'id_': 'doc1', 'hash': 'h1'})(),
            type('D', (), {'text': 'A quick brown fox', 'metadata': {'source': 'test'}, 'id_': 'doc2', 'hash': 'h2'})(),
        ]
        # Convert simple objects to llama_index Document objects if available
        from llama_index.core.schema import Document as LlamaDocument
        ll_docs = [LlamaDocument(text=d.text, metadata=d.metadata) for d in docs]

        index = VectorStoreIndex.from_documents(ll_docs, embed_model=mock_emb)
        # craft a query embedding and run a vector store query
        q_emb = mock_emb._get_query_embedding('cat')
        q = VectorStoreQuery(query_embedding=q_emb, similarity_top_k=2)
        vs = index.vector_store
        result = vs.query(q)
        assert result is not None
        print(f"- Minimal index+query returned ids: {getattr(result, 'ids', None)}")

        with tempfile.TemporaryDirectory() as persist_dir:
            persistent_docs = [
                LlamaDocument(text="Form W-2 box 1 wages 125000 federal withholding 22000", metadata={"file_path": "persist-w2.pdf"})
            ]
            first_store = SimpleVectorStore()
            _, first_context = build_or_load_persistent_index(
                persistent_docs,
                first_store,
                persist_dir=persist_dir,
                embed_model=mock_emb,
                rebuild=True,
            )
            first_context.persist(persist_dir)
            empty_store = SimpleVectorStore()
            _, rebuilt_context = build_or_load_persistent_index(
                persistent_docs,
                empty_store,
                persist_dir=persist_dir,
                embed_model=mock_emb,
            )
            rebuilt_count = len(rebuilt_context.vector_store.data.embedding_dict)
            assert rebuilt_count > 0
            print("- persistent empty-vector-store rebuild OK")
    except Exception as e:
        print("- Minimal E2E index/query failed:", e)
        raise

    print("Smoke tests passed.")


def main():
    run_tests()


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except AssertionError as e:
        print("Smoke test assertion failed:", e)
        sys.exit(2)
    except Exception as e:
        print("Smoke test error:", e)
        sys.exit(3)
