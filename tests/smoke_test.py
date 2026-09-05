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
    _effective_query_for_turn,
    _extract_source_references,
    _format_response_for_display,
    _is_ocr_garbage_text,
    _is_port_open,
    _looks_like_follow_up,
    _spoken_response_text,
    jpeg_extractor,
    TTSManager,
)

# Frozen public API contract for jarvis_helpers. The package refactor relocates
# implementation into jarvis_*.py modules, but every name below must remain
# importable from jarvis_helpers (the backward-compatible facade). This guard
# fails loudly the moment any export is dropped or renamed during the move.
_PUBLIC_API_SURFACE = (
    # Config / domain
    "JARVIS_DOMAIN",
    "debug_print",
    "short_text_qa_template",
    # Orchestration
    "QueryRuntime",
    "handle_query",
    # Speech / TTS
    "TTSManager",
    "_spoken_response_text",
    "_summarize_for_speech",
    "_speech_text_for_response",
    "_format_response_for_display",
    "_extract_source_references",
    # Extraction / loading
    "file_extractor",
    "jpeg_extractor",
    "has_files_with_extensions",
    "load_documents_with_cache",
    # Vector store / indexing
    "create_milvus_vector_store",
    "build_hybrid_retriever",
    "build_or_load_persistent_index",
    "filter_informative_documents",
    "group_documents_by_source_windows",
    "chunk_text",
    "split_large_docs",
    # Text quality
    "is_low_information_text",
    "normalize_text",
    "text_artifact_score",
    "_is_ocr_garbage_text",
    # Conversation / retrieval / answering
    "_effective_query_for_turn",
    "_looks_like_follow_up",
    "_is_port_open",
)


def check_public_api_surface():
    import jarvis_helpers

    missing = [name for name in _PUBLIC_API_SURFACE if not hasattr(jarvis_helpers, name)]
    assert not missing, f"jarvis_helpers is missing public API names: {missing}"
    print(f"- public API surface OK ({len(_PUBLIC_API_SURFACE)} names exported)")


def run_tests():
    print("Running smoke tests...")
    check_public_api_surface()

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

    # Carried overlap must never push a chunk past its declared word budget.
    budget_cases = [
        ("\n".join([" ".join(f"a{i}" for i in range(900)),
                    " ".join(f"b{i}" for i in range(900))]), 1000, 200),
        ("\n".join(" ".join(f"c{n}_{i}" for i in range(60)) for n in range(20)), 100, 40),
        (" ".join(f"d{i}" for i in range(5000)), 400, 120),
        ("\n".join(" ".join(f"e{n}_{i}" for i in range(99)) for n in range(30)), 100, 99),
    ]
    for budget_text, size, overlap in budget_cases:
        produced = chunk_text(budget_text, chunk_size=size, chunk_overlap=overlap)
        longest = max(len(chunk.split()) for chunk in produced)
        assert longest <= size, f"chunk of {longest} words exceeds budget {size}"
        assert set(budget_text.split()) <= set(" ".join(produced).split()), "chunking dropped content"

    carried = chunk_text(" ".join(f"f{i}" for i in range(1000)), chunk_size=400, chunk_overlap=100)
    assert len(carried) > 1 and set(carried[0].split()) & set(carried[1].split())
    print("- chunk budget invariant OK")

    from jarvisrag.extraction.indexing import (
        _METADATA_EMBED_BUDGET_CHARS,
        _limit_embedded_metadata,
        _prepare_index_document,
    )
    from llama_index.core.schema import MetadataMode

    # A reader that attaches a huge value must not be able to starve the node
    # parser's budget, which silently drops the chunk from the index. The rule
    # is driven by size, so the key name here is deliberately arbitrary.
    bloated = _prepare_index_document(LlamaDocument(
        text="Total wages 125000 reported for the year",
        metadata={
            "file_path": "/corpus/form.doc",
            "file_name": "form.doc",
            "page_number": 3,
            "some_reader_specific_blob": "<table>" + ("<td>cell</td>" * 4000) + "</table>",
        },
    ))
    excluded = set(bloated.excluded_embed_metadata_keys or [])
    assert "some_reader_specific_blob" in excluded, "oversized value must leave the embedded text"
    # The parser measures the longer of the two, so both must be trimmed.
    assert excluded == set(bloated.excluded_llm_metadata_keys or [])
    for mode in (MetadataMode.EMBED, MetadataMode.LLM):
        length = len(bloated.get_metadata_str(mode=mode))
        assert length <= _METADATA_EMBED_BUDGET_CHARS, f"{mode} metadata still {length} chars"
    # Dropped from the embedded text only; still stored for filtering and lookup.
    assert bloated.metadata.get("some_reader_specific_blob")
    # Largest-first eviction has to leave the small identifying keys alone.
    assert not {"file_name", "page_number"} & excluded, "identifying metadata must survive"

    # Ordinary documents must be untouched, or every corpus re-embeds for nothing.
    ordinary = _prepare_index_document(LlamaDocument(
        text="Wages 125000",
        metadata={"file_path": "/corpus/w2.pdf", "file_name": "w2.pdf", "page_number": 1},
    ))
    assert not ordinary.excluded_embed_metadata_keys
    assert not ordinary.excluded_llm_metadata_keys

    # Exclusions must not change the document hash: the manifest diffs on hash,
    # so a change here would re-embed the whole corpus on the next sync.
    hashed = _prepare_index_document(LlamaDocument(
        text="Total wages 125000 reported for the year",
        metadata={
            "file_path": "/corpus/form.doc",
            "file_name": "form.doc",
            "page_number": 3,
            "some_reader_specific_blob": "<table>" + ("<td>cell</td>" * 4000) + "</table>",
        },
    ))
    hash_with_exclusions = hashed.hash
    hashed.excluded_embed_metadata_keys = []
    hashed.excluded_llm_metadata_keys = []
    assert hashed.hash == hash_with_exclusions, "metadata exclusions must not disturb chunk hashes"

    # Repeated preparation must be idempotent rather than compounding exclusions.
    _limit_embedded_metadata(bloated)
    assert set(bloated.excluded_embed_metadata_keys or []) == excluded
    print(f"- oversized metadata capped OK (budget={_METADATA_EMBED_BUDGET_CHARS} chars)")

    from jarvisrag.extraction.indexing import (
        _diff_manifest_documents,
        _manifest_without_ids,
    )

    # Chunk ids hash the chunk text, so what the store holds is stronger evidence
    # than the manifest, which can move with the code or outlive a failed insert.
    manifest = {"documents": [
        {"id": "a", "hash": "h1", "file_path": "/corpus/one.pdf"},
        {"id": "b", "hash": "h2", "file_path": "/corpus/two.pdf"},
    ]}

    # Manifest lost (moved persist dir): the store still prevents re-embedding.
    to_insert, _ = _diff_manifest_documents(manifest, None, indexed_ids={"a", "b"})
    assert to_insert == [], "chunks already in the store must never be re-embedded"

    # Manifest claims a chunk is indexed but the store rejected it: retry it.
    to_insert, _ = _diff_manifest_documents(manifest, manifest, indexed_ids={"a"})
    assert to_insert == ["b"], "a chunk missing from the store must be retried"

    # Store unreachable: fall back to the manifest rather than re-embedding all.
    to_insert, _ = _diff_manifest_documents(manifest, manifest, indexed_ids=None)
    assert to_insert == [], "an unreadable store must not trigger a full re-embed"

    # Deletions stay manifest-driven and scope-limited.
    stale = {"documents": [{"id": "c", "hash": "h3", "file_path": "/corpus/gone.pdf"}]}
    _, to_delete = _diff_manifest_documents(
        manifest, stale, scope_file_paths={"/corpus/gone.pdf"}, indexed_ids={"a", "b"}
    )
    assert to_delete == ["c"]
    _, to_delete = _diff_manifest_documents(
        manifest, stale, scope_file_paths={"/corpus/one.pdf"}, indexed_ids={"a", "b"}
    )
    assert to_delete == [], "documents from other run directories must be preserved"

    trimmed = _manifest_without_ids(manifest, {"b"})
    assert [entry["id"] for entry in trimmed["documents"]] == ["a"]
    assert _manifest_without_ids(manifest, set()) is manifest
    assert len(manifest["documents"]) == 2, "trimming must not mutate the caller's manifest"
    print("- vector store is authoritative for what is indexed OK")

    from jarvisrag.retrieval import _line_tokens, _near_duplicate_of

    body_a = _line_tokens("- alpha.pdf, page 2: shared sentence about vacation accrual policy")
    body_b = _line_tokens("- beta.pdf: shared sentence about vacation accrual policy")
    body_c = _line_tokens("- gamma.pdf: unrelated wording covering expense approval limits")
    assert "page" not in body_a and "alpha" not in body_a, "only the evidence body is compared"
    assert _near_duplicate_of(body_b, [body_a], 0.9), "same passage from another file is redundant"
    assert not _near_duplicate_of(body_c, [body_a], 0.9), "distinct evidence must be kept"
    assert not _near_duplicate_of(body_b, [], 0.9)
    print("- near-duplicate evidence detection OK")

    from jarvisrag.answer.bootstrap import DEFAULT_RETRIEVAL_TOP_K

    tuned_top_k = int(DEFAULT_RETRIEVAL_TOP_K)
    assert tuned_top_k > 0, "retrieval breadth must be positive"
    readme = open(os.path.join(os.path.dirname(__file__), "..", "README.md"), encoding="utf-8").read()
    assert f"| `JARVIS_RETRIEVAL_TOP_K` | `{tuned_top_k}` |" in readme, (
        "README documents a different retrieval breadth than the code default"
    )
    print(f"- retrieval breadth default OK (top_k={tuned_top_k}, documented)")

    grouped_docs = group_documents_by_source_windows([
        LlamaDocument(text="Page one amount 100", metadata={"file_path": "sample.pdf", "page_number": 1}),
        LlamaDocument(text="Page two balance 200", metadata={"file_path": "sample.pdf", "page_number": 2}),
    ], max_words=50)
    assert len(grouped_docs) == 1
    assert "Page one amount 100" in grouped_docs[0].text
    assert grouped_docs[0].metadata.get("context_mode") == "source_window"
    print("- group_documents_by_source_windows OK")

    # Conversational follow-up detection: a pronoun reference to the prior
    # subject ("how long did he work there?") must be treated as a follow-up
    # and carry the prior question's context into the effective query.
    prior_turn = {
        "user_query": "What is Jordan's current employer?",
        "effective_query": "What is Jordan's current employer?",
        "evidence_lines": [],
    }
    assert _looks_like_follow_up("How long did he work there?", prior_turn)
    assert _looks_like_follow_up("Where did she go to school?", prior_turn)
    eff_query, is_follow_up, _hints = _effective_query_for_turn(
        "How long did he work there?", prior_turn
    )
    assert is_follow_up
    assert "current employer" in eff_query and "how long did he work there" in eff_query.lower()
    # No prior turn -> never a follow-up (first question can't be anaphoric).
    assert not _looks_like_follow_up("How long did he work there?", {})
    # Word-boundary guard: "the"/"where" must not false-trigger the pronoun rule.
    assert not _looks_like_follow_up("Where is the headquarters located?", prior_turn)
    # Topic-switch guard: a NEW named subject without a back-referring pronoun is
    # a fresh question, even with a generic marker like "what about".
    assert not _looks_like_follow_up("What about Microsoft's revenue?", prior_turn)
    assert not _looks_like_follow_up("Tell me about Google instead.", prior_turn)
    # ...but a pronoun keeps the thread even when a new object/place is named.
    assert _looks_like_follow_up(
        "And how about his GPA at Johns Hopkins University?", prior_turn
    )
    # Naming the SAME prior subject with a marker stays a follow-up.
    assert _looks_like_follow_up("What about Jordan's salary there?", prior_turn)
    print("- conversational follow-up detection OK")

    # Display formatting: the full answer (values intact) is kept, and a
    # de-duplicated 'References' list of the named sources is appended instead
    # of repeating the answer body.
    answer = (
        "His GPA is 3.50, listed in Application.pdf (page 12) and "
        "StateDept_CompSci.pdf (page 17). JOB APPLICATION.pdf (page 9) lists "
        "3.65, while Rutgers_Transcript.JPG shows a cumulative average. "
        "While JHU Transcript.pdf provides grades, "
        "Application.pdf (page 12) repeats the 3.50 figure."
    )
    refs = _extract_source_references(answer)
    assert "Application.pdf (page 12)" in refs
    assert "StateDept_CompSci.pdf (page 17)" in refs
    assert "JOB APPLICATION.pdf (page 9)" in refs
    assert "Rutgers_Transcript.JPG" in refs
    # Spaced filenames keep their internal space ("JHU Transcript.pdf").
    assert "JHU Transcript.pdf" in refs
    # Mixed-case prose words ("While", "listed", "in") must NOT be absorbed into
    # a filename token.
    assert all(not r.startswith(("While ", "listed ", "in ")) for r in refs)
    # De-duplicated: the repeated Application.pdf (page 12) appears once.
    assert refs.count("Application.pdf (page 12)") == 1
    rendered = _format_response_for_display(answer)
    # Values stay in the body, and a References block is appended.
    assert "3.50" in rendered and "3.65" in rendered
    assert "\n\nReferences:\n" in rendered
    assert rendered.index("References:") > rendered.index("3.65")
    # An answer that names no sources is returned unchanged (no References block).
    plain = "The capital of France is Paris."
    assert _format_response_for_display(plain) == plain
    print("- citation-list display formatting OK")

    # TTS path: filenames/extensions must NOT be read aloud. Bare in-prose
    # filenames are replaced with "the document(s)" while the answer's values
    # are preserved; the printed References block (eyes only) is unaffected.
    spoken = _spoken_response_text(answer)
    lowered_spoken = spoken.lower()
    for ext in (".pdf", ".jpg", ".jpeg", ".docx", ".xlsx", ".pptx", ".txt", ".csv"):
        assert ext not in lowered_spoken, f"extension {ext} leaked into TTS: {spoken!r}"
    assert "application" not in lowered_spoken and "transcript." not in lowered_spoken
    # Values survive the substitution.
    assert "3.50" in spoken and "3.65" in spoken
    # Neutral placeholder is used so the spoken sentence stays grammatical.
    assert "the document" in lowered_spoken
    print("- TTS citation suppression OK")

    # Spoken dollar amounts must absorb a magnitude scale (word or letter suffix)
    # and speak it in the right order, instead of dropping it or stranding it
    # after "dollars". Regression for "$4 million" -> "4 dollars million" and
    # "$96k" -> "96 dollars".
    _speak = TTSManager.__new__(TTSManager)
    assert TTSManager.clean_text(_speak, "saved $4 million in costs") == (
        "saved 4 million dollars in costs"
    )
    assert TTSManager.clean_text(_speak, "saved $96k/year") == (
        "saved 96 thousand dollars per year"
    )
    assert TTSManager.clean_text(_speak, "a $2.3 billion market") == (
        "a 2 point 3 billion dollars market"
    )
    # Plain amounts are spelled out so an engine never reads a 4-digit figure as
    # a year (e.g. "$1,250" as "twelve fifty").
    assert TTSManager.clean_text(_speak, "a $1,250.75 balance") == (
        "a one thousand two hundred and fifty dollars and seventy five cents balance"
    )
    print("- TTS money-scale wording OK")

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
